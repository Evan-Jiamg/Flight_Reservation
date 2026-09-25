#!/usr/bin/env python3
"""File-based GPU job queue for ONE host. Pure python + subprocess; all state lives in files.

Layout of a queue directory DIR:
  jobs/<id>.json         one file per job: spec + state (atomic writes: tmp + fsync + rename)
  logs/<id>.a<N>.log     stdout+stderr of attempt N
  hb/<id>.hb             heartbeat, touched every JOBQ_HB_SECONDS (default 60) by the wrapper
  results/<id>.a<N>.json exit code of attempt N, written atomically by the wrapper
  STOP                   present -> the worker launches nothing (written on an API quota error)
  status.json            summary (counts, per-job state, heartbeat age, attempts)
  worker.lock, worker.log

Commands
  jobq.py add    --queue DIR --id ID --cmd "..." --gpu-mib N [--allowed-gpus 1,2] [--retries 3] [--after A,B]
  jobq.py worker --queue DIR --allowed-gpus 1[,2] [--max-running 4] [--poll 15] [--exit-when-idle]
                 [--smi-cmd "nvidia-smi ..."] [--reserve-seconds 900]
  jobq.py status --queue DIR [--json]
  jobq.py resume --queue DIR               remove STOP, QUOTA jobs -> PENDING
  jobq.py reset  --queue DIR --id ID       FAILED/QUOTA job -> PENDING with fresh counters

Job life cycle
  PENDING -> (deps DONE, not_before passed, an allowed GPU has >= gpu_mib free) -> RUNNING
  RUNNING -> exit 0                                   -> DONE
          -> exit != 0, API quota pattern in the log  -> QUOTA, STOP file written, nothing new starts
          -> exit != 0, CUDA OOM pattern              -> PENDING with backoff (counts as a failure)
          -> other exit != 0                          -> PENDING with short backoff, FAILED after --retries
          -> wrapper pid dead and no result file      -> PENDING (counted as 'lost'; FAILED after retries+3)
GPU placement: allowed = worker --allowed-gpus INTERSECT job --allowed-gpus (GPU 0 therefore only when the
worker lists it). The job gets CUDA_VISIBLE_DEVICES=<gpu>, so inside the job the device is cuda:0;
"{gpu}" in --cmd expands to 0 (the local index) and "{phys_gpu}" to the physical index. Free memory =
nvidia-smi memory.free minus gpu_mib of jobs this worker started on that GPU within --reserve-seconds
(model loading takes a while to show up in nvidia-smi). A gpu_mib of 0 means a CPU/API job:
CUDA_VISIBLE_DEVICES="" and no GPU check.
Each job runs as: setsid python jobq.py _wrap ... ; the wrapper runs the command with bash, touches the
heartbeat and writes the result file. The wrapper survives a worker crash; a restarted worker adopts it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

STATES = ("PENDING", "RUNNING", "DONE", "FAILED", "QUOTA")
OOM_RE = re.compile(r"CUDA out of memory|torch\.(cuda\.)?OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|"
                    r"cudaErrorMemoryAllocation|CUDA error: out of memory")
QUOTA_RE = re.compile(r"insufficient_quota|exceeded your current quota|Error code: 402\b|"
                      r"\b402 Payment Required|status[_ ]?code[\"']?\s*[=:]\s*402\b|HTTP/[0-9.]+\"? 402\b|"
                      r"PaymentRequired", re.I)
LOG_TAIL_BYTES = 256 * 1024
IS_POSIX = os.name == "posix"
HERE = os.path.abspath(__file__)


# ------------------------------------------------------------------ file helpers
def atomic_write_json(path, obj):
    tmp = "%s.tmp.%d.%d" % (path, os.getpid(), threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def touch(path):
    with open(path, "a"):
        pass
    os.utime(path, None)


def log_tail(path, n=LOG_TAIL_BYTES):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def classify(exit_code, log_text):
    """-> 'DONE' | 'QUOTA' | 'OOM' | 'FAIL'."""
    if exit_code == 0:
        return "DONE"
    if QUOTA_RE.search(log_text):
        return "QUOTA"
    if OOM_RE.search(log_text):
        return "OOM"
    return "FAIL"


# ------------------------------------------------------------------ process helpers
def proc_start_time(pid):
    """Linux start time (clock ticks since boot) to detect pid reuse; None elsewhere."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            s = f.read()
        return int(s[s.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def pid_alive(pid, start_time=None):
    if not pid:
        return False
    if IS_POSIX:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        try:  # a zombie is dead for our purposes
            with open("/proc/%d/stat" % pid) as f:
                s = f.read()
            if s[s.rfind(")") + 2:].split()[0] == "Z":
                return False
        except OSError:
            pass
        if start_time is not None:
            st = proc_start_time(pid)
            if st is not None and st != start_time:
                return False
        return True
    import ctypes  # Windows (local tests only)
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259


def kill_group(pgid):
    if IS_POSIX and pgid:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass


# ------------------------------------------------------------------ queue
class Queue:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        for d in ("jobs", "logs", "hb", "results"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)

    def p(self, *a):
        return os.path.join(self.root, *a)

    def job_path(self, jid):
        return self.p("jobs", "%s.json" % jid)

    def load(self, jid):
        return read_json(self.job_path(jid))

    def save(self, job):
        job["updated"] = time.time()
        atomic_write_json(self.job_path(job["id"]), job)

    def all_jobs(self):
        out = []
        for n in os.listdir(self.p("jobs")):
            if n.endswith(".json"):
                j = read_json(self.p("jobs", n))
                if j:
                    out.append(j)
        return sorted(out, key=lambda j: (j.get("seq", 0), j["id"]))

    def stopped(self):
        return os.path.exists(self.p("STOP"))

    def log(self, msg):
        line = "%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        with open(self.p("worker.log"), "a", encoding="utf-8") as f:
            f.write(line)

    def add(self, jid, cmd, gpu_mib, allowed_gpus=None, retries=3, after=None, env=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", jid):
            raise ValueError("job id must match [A-Za-z0-9_.-]+: %r" % jid)
        if os.path.exists(self.job_path(jid)):
            raise ValueError("job %s already exists" % jid)
        seq = int(time.time() * 1e6)
        job = {"id": jid, "cmd": cmd, "gpu_mib": int(gpu_mib), "allowed_gpus": allowed_gpus,
               "retries": int(retries), "after": list(after or []), "env": env or {},
               "state": "PENDING", "attempts": 0, "failures": 0, "lost": 0, "not_before": 0,
               "pid": None, "pid_start": None, "gpu": None, "exit_code": None, "reason": None,
               "history": [], "seq": seq, "created": time.time()}
        self.save(job)
        return job

    # -------------------------------------------------------------- status
    def summary(self):
        jobs = self.all_jobs()
        by_id = {j["id"]: j for j in jobs}
        now = time.time()
        counts = {s: 0 for s in STATES}
        per = {}
        for j in jobs:
            counts[j["state"]] = counts.get(j["state"], 0) + 1
            hb = self.p("hb", "%s.hb" % j["id"])
            age = round(now - os.path.getmtime(hb), 1) if os.path.exists(hb) else None
            blocked = [d for d in j["after"] if (by_id.get(d) or {}).get("state") in ("FAILED", "QUOTA")
                       or d not in by_id]
            per[j["id"]] = {"state": j["state"], "attempts": j["attempts"], "failures": j["failures"],
                            "lost": j.get("lost", 0), "gpu": j.get("gpu"), "pid": j.get("pid"),
                            "last_heartbeat_age_s": age, "exit_code": j.get("exit_code"),
                            "reason": j.get("reason"), "after": j["after"], "blocked_by": blocked,
                            "log": j.get("log"), "gpu_mib": j["gpu_mib"]}
        return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stopped": self.stopped(),
                "stop_reason": (open(self.p("STOP")).read().strip() if self.stopped() else None),
                "counts": counts, "jobs": per}

    def write_status(self):
        s = self.summary()
        atomic_write_json(self.p("status.json"), s)
        return s


def parse_smi(text):
    free = {}
    for line in text.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            free[int(parts[0])] = int(float(parts[1]))
    return free


def smi_query(cmd):
    def q():
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise RuntimeError("nvidia-smi failed: %s" % out.stderr.strip()[:200])
        return parse_smi(out.stdout)
    return q


DEFAULT_SMI = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"


class Worker:
    def __init__(self, queue, allowed_gpus, max_running=4, gpu_query=None, reserve_seconds=900,
                 oom_backoff=(120, 1800), fail_backoff=30, python=sys.executable):
        self.q = queue
        self.allowed = [int(g) for g in allowed_gpus]
        self.max_running = max_running
        self.gpu_query = gpu_query or smi_query(DEFAULT_SMI)
        self.reserve_seconds = reserve_seconds
        self.oom_backoff, self.fail_backoff = oom_backoff, fail_backoff
        self.python = python
        self.children = {}          # jid -> Popen (for reaping)

    def start(self):
        """Take the queue lock, then requeue RUNNING jobs whose wrapper died (crash recovery)."""
        self.acquire_lock()
        self.recover()

    # -------------------------------------------------------------- lock
    def acquire_lock(self):
        path = self.q.p("worker.lock")
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"pid": os.getpid(), "start": proc_start_time(os.getpid())}).encode())
                os.close(fd)
                return
            except FileExistsError:
                info = read_json(path, {}) or {}
                if pid_alive(info.get("pid"), info.get("start")):
                    raise SystemExit("another worker (pid %s) owns %s" % (info.get("pid"), self.q.root))
                os.remove(path)
        raise SystemExit("cannot take worker lock")

    def release_lock(self):
        try:
            os.remove(self.q.p("worker.lock"))
        except OSError:
            pass

    # -------------------------------------------------------------- recovery
    def recover(self):
        """On (re)start: RUNNING jobs whose wrapper is dead and left no result are requeued."""
        for j in self.q.all_jobs():
            if j["state"] != "RUNNING":
                continue
            if os.path.exists(self.result_path(j)):
                continue                           # finished while no worker watched; collected next tick
            if not pid_alive(j.get("pid"), j.get("pid_start")):
                kill_group(j.get("pgid"))
                self.requeue_lost(j, "wrapper pid %s dead at worker start" % j.get("pid"))

    def requeue_lost(self, j, why):
        j["lost"] = j.get("lost", 0) + 1
        j["history"].append({"attempt": j["attempts"], "outcome": "LOST", "why": why, "t": time.time()})
        if j["lost"] > j["retries"] + 3:
            j.update(state="FAILED", reason="lost %d times" % j["lost"], pid=None)
        else:
            j.update(state="PENDING", pid=None, gpu=None, reason="requeued: " + why)
        self.q.save(j)
        self.q.log("%s %s -> %s" % (j["id"], why, j["state"]))

    def result_path(self, j):
        return self.q.p("results", "%s.a%d.json" % (j["id"], j["attempts"]))

    # -------------------------------------------------------------- one scheduling round
    def tick(self):
        now = time.time()
        jobs = self.q.all_jobs()
        running = [j for j in jobs if j["state"] == "RUNNING"]
        for j in running:
            self.check_running(j)
        jobs = self.q.all_jobs()
        by_id = {j["id"]: j for j in jobs}
        running = [j for j in jobs if j["state"] == "RUNNING"]
        if self.q.stopped():
            self.q.write_status()
            return {"running": len(running), "launched": 0, "stopped": True}
        launched = 0
        free = None
        for j in jobs:
            if len(running) + launched >= self.max_running:
                break
            if j["state"] != "PENDING" or j.get("not_before", 0) > now:
                continue
            if any((by_id.get(d) or {}).get("state") != "DONE" for d in j["after"]):
                continue
            gpu = None
            if j["gpu_mib"] > 0:
                if free is None:
                    try:
                        free = self.effective_free(jobs)
                    except Exception as e:  # noqa: BLE001 - no GPU info: launch no GPU job this round
                        self.q.log("gpu query failed: %s" % e)
                        free = {}
                ok = self.gpus_for(j)
                cands = [(free.get(g, -1), g) for g in ok if free.get(g, -1) >= j["gpu_mib"]]
                if not cands:
                    continue
                gpu = max(cands)[1]
                free[gpu] -= j["gpu_mib"]
            self.launch(j, gpu)
            launched += 1
        self.q.write_status()
        return {"running": len(running) + launched, "launched": launched, "stopped": False}

    def gpus_for(self, j):
        ok = list(self.allowed)
        if j.get("allowed_gpus"):
            ok = [g for g in ok if g in j["allowed_gpus"]]
        return ok

    def effective_free(self, jobs):
        free = {g: v for g, v in self.gpu_query().items() if g in self.allowed}
        now = time.time()
        for j in jobs:
            if j["state"] == "RUNNING" and j.get("gpu") is not None and now - (j.get("started") or 0) < self.reserve_seconds:
                if j["gpu"] in free:
                    free[j["gpu"]] -= j["gpu_mib"]
        return free

    def launch(self, j, gpu):
        j["attempts"] += 1
        a = j["attempts"]
        logp = self.q.p("logs", "%s.a%d.log" % (j["id"], a))
        for stale in (self.result_path(j),):
            if os.path.exists(stale):
                os.remove(stale)
        env = dict(os.environ)
        env.update(j.get("env") or {})
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        env["JOBQ_GPU"] = "" if gpu is None else str(gpu)
        env["JOBQ_JOB_ID"], env["JOBQ_ATTEMPT"] = j["id"], str(a)
        argv = [self.python, HERE, "_wrap", "--queue", self.q.root, "--id", j["id"], "--attempt", str(a),
                "--phys-gpu", "" if gpu is None else str(gpu)]
        kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
              "env": env, "cwd": os.getcwd()}
        if IS_POSIX:
            kw["start_new_session"] = True      # setsid: survives the worker, own process group
        else:
            kw["creationflags"] = 0x00000200    # CREATE_NEW_PROCESS_GROUP
        pr = subprocess.Popen(argv, **kw)
        self.children[j["id"]] = pr
        j.update(state="RUNNING", pid=pr.pid, pgid=pr.pid if IS_POSIX else None,
                 pid_start=proc_start_time(pr.pid), gpu=gpu, started=time.time(), log=logp,
                 exit_code=None, reason=None)
        self.q.save(j)
        self.q.log("launch %s attempt %d gpu %s pid %d" % (j["id"], a, gpu, pr.pid))

    def check_running(self, j):
        pr = self.children.get(j["id"])
        if pr is not None:
            pr.poll()                            # reap
        res = read_json(self.result_path(j))
        if res is None:
            alive = pr.poll() is None if pr is not None else pid_alive(j.get("pid"), j.get("pid_start"))
            if not alive:
                time.sleep(0.2)                  # the result may be landing right now
                res = read_json(self.result_path(j))
                if res is None:
                    kill_group(j.get("pgid"))
                    self.children.pop(j["id"], None)
                    self.requeue_lost(j, "wrapper pid %s exited without a result" % j.get("pid"))
            if res is None:
                return
        self.children.pop(j["id"], None)
        self.finish(j, int(res["exit_code"]))

    def finish(self, j, code):
        kind = classify(code, log_tail(j.get("log") or ""))
        j["exit_code"], j["ended"], j["pid"] = code, time.time(), None
        j["history"].append({"attempt": j["attempts"], "outcome": kind, "exit_code": code, "gpu": j.get("gpu"),
                             "t": time.time()})
        if kind == "DONE":
            j.update(state="DONE", reason=None)
        elif kind == "QUOTA":
            j.update(state="QUOTA", reason="API quota / payment error; queue stopped")
            atomic_stop(self.q, "job %s hit an API quota error (exit %d) at %s" % (
                j["id"], code, time.strftime("%Y-%m-%d %H:%M:%S")))
        else:
            j["failures"] += 1
            if j["failures"] > j["retries"]:
                j.update(state="FAILED", reason="%s after %d failures" % (kind, j["failures"]))
            else:
                if kind == "OOM":
                    lo, hi = self.oom_backoff
                    delay = min(hi, lo * 2 ** (j["failures"] - 1))
                else:
                    delay = self.fail_backoff
                j.update(state="PENDING", not_before=time.time() + delay,
                         reason="%s (exit %d); retry in %ds" % (kind, code, delay))
        self.q.save(j)
        self.q.log("finish %s attempt %d exit %d -> %s (%s)" % (j["id"], j["attempts"], code, j["state"], kind))

    def idle(self):
        jobs = self.q.all_jobs()
        by_id = {j["id"]: j for j in jobs}
        if any(j["state"] == "RUNNING" for j in jobs):
            return False
        if self.q.stopped():
            return True

        def runnable(j):  # could still run some day (deps not permanently failed)
            return all((by_id.get(d) or {}).get("state") in ("DONE", "PENDING", "RUNNING") for d in j["after"])
        return not any(j["state"] == "PENDING" and runnable(j) for j in jobs)


def atomic_stop(q, text):
    tmp = q.p("STOP.tmp")
    with open(tmp, "w") as f:
        f.write(text + "\n")
    os.replace(tmp, q.p("STOP"))
    q.log("STOP: " + text)


# ------------------------------------------------------------------ wrapper (runs detached)
def wrap(queue_dir, jid, attempt, phys_gpu):
    q = Queue(queue_dir)
    j = q.load(jid)
    logp = q.p("logs", "%s.a%d.log" % (jid, attempt))
    hbp = q.p("hb", "%s.hb" % jid)
    hb_s = float(os.environ.get("JOBQ_HB_SECONDS", "60"))
    cmd = j["cmd"].replace("{phys_gpu}", phys_gpu).replace("{gpu}", "0" if phys_gpu != "" else "")
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            try:
                touch(hbp)
            except OSError:
                pass
            stop.wait(hb_s)
    th = threading.Thread(target=beat, daemon=True)
    th.start()
    with open(logp, "ab") as lf:
        lf.write(("[jobq] %s attempt %d gpu %s start %s\n[jobq] cmd: %s\n" % (
            jid, attempt, phys_gpu or "-", time.strftime("%Y-%m-%d %H:%M:%S"), cmd)).encode())
        lf.flush()
        kw = {"shell": True, "stdout": lf, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
        if IS_POSIX:
            kw["executable"] = "/bin/bash"
        try:
            pr = subprocess.Popen(cmd, **kw)

            def term(*_):
                try:
                    pr.terminate()
                except OSError:
                    pass
            if IS_POSIX:
                signal.signal(signal.SIGTERM, term)
            code = pr.wait()
        except Exception as e:  # noqa: BLE001
            lf.write(("[jobq] wrapper error: %r\n" % (e,)).encode())
            code = 127
        lf.write(("[jobq] exit %d at %s\n" % (code, time.strftime("%Y-%m-%d %H:%M:%S"))).encode())
    stop.set()
    touch(hbp)
    atomic_write_json(q.p("results", "%s.a%d.json" % (jid, attempt)), {"exit_code": code, "ended": time.time()})
    return code


# ------------------------------------------------------------------ CLI
def ints(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def print_status(s, as_json=False):
    if as_json:
        print(json.dumps(s, indent=1))
        return
    print("queue %s  stopped=%s%s" % (s["time"], s["stopped"], ("  (%s)" % s["stop_reason"]) if s["stopped"] else ""))
    print("counts: " + "  ".join("%s=%d" % kv for kv in s["counts"].items()))
    print("%-28s %-8s %4s %4s %4s %4s %8s %s" % ("id", "state", "att", "fail", "gpu", "exit", "hb_age", "note"))
    for jid, j in s["jobs"].items():
        note = j["reason"] or ""
        if j["blocked_by"] and j["state"] == "PENDING":
            note = "BLOCKED by %s" % ",".join(j["blocked_by"])
        print("%-28s %-8s %4d %4d %4s %4s %8s %s" % (
            jid[:28], j["state"], j["attempts"], j["failures"], "-" if j["gpu"] is None else j["gpu"],
            "-" if j["exit_code"] is None else j["exit_code"],
            "-" if j["last_heartbeat_age_s"] is None else "%.0fs" % j["last_heartbeat_age_s"], note[:60]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add")
    a.add_argument("--queue", required=True)
    a.add_argument("--id", required=True)
    a.add_argument("--cmd", required=True)
    a.add_argument("--gpu-mib", type=int, required=True)
    a.add_argument("--allowed-gpus", default="")
    a.add_argument("--retries", type=int, default=3)
    a.add_argument("--after", default="")
    a.add_argument("--env", action="append", default=[], help="KEY=VALUE (repeatable)")
    w = sub.add_parser("worker")
    w.add_argument("--queue", required=True)
    w.add_argument("--allowed-gpus", required=True)
    w.add_argument("--max-running", type=int, default=4)
    w.add_argument("--poll", type=float, default=15)
    w.add_argument("--reserve-seconds", type=float, default=900)
    w.add_argument("--smi-cmd", default=os.environ.get("JOBQ_SMI_CMD", DEFAULT_SMI))
    w.add_argument("--exit-when-idle", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--queue", required=True)
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("resume")
    r.add_argument("--queue", required=True)
    rs = sub.add_parser("reset")
    rs.add_argument("--queue", required=True)
    rs.add_argument("--id", required=True)
    wr = sub.add_parser("_wrap")
    wr.add_argument("--queue", required=True)
    wr.add_argument("--id", required=True)
    wr.add_argument("--attempt", type=int, required=True)
    wr.add_argument("--phys-gpu", default="")
    x = ap.parse_args(argv)

    if x.action == "_wrap":
        return wrap(x.queue, x.id, x.attempt, x.phys_gpu)
    q = Queue(x.queue)
    if x.action == "add":
        env = dict(kv.split("=", 1) for kv in x.env)
        j = q.add(x.id, x.cmd, x.gpu_mib, ints(x.allowed_gpus), x.retries,
                  [d for d in x.after.split(",") if d.strip()], env)
        q.write_status()
        print("added %s (gpu_mib %d, after %s)" % (j["id"], j["gpu_mib"], j["after"] or "-"))
        return 0
    if x.action == "status":
        print_status(q.write_status(), x.json)
        return 0
    if x.action == "resume":
        if os.path.exists(q.p("STOP")):
            os.remove(q.p("STOP"))
        for j in q.all_jobs():
            if j["state"] == "QUOTA":
                j.update(state="PENDING", reason="resumed after quota stop", not_before=0)
                q.save(j)
        q.write_status()
        print("resumed")
        return 0
    if x.action == "reset":
        j = q.load(x.id)
        if not j:
            raise SystemExit("no job %s" % x.id)
        if j["state"] not in ("FAILED", "QUOTA", "PENDING"):
            raise SystemExit("job %s is %s; only FAILED/QUOTA/PENDING can be reset" % (x.id, j["state"]))
        j.update(state="PENDING", failures=0, lost=0, not_before=0, reason="reset")
        q.save(j)
        q.write_status()
        print("reset %s" % x.id)
        return 0
    # worker
    wk = Worker(q, ints(x.allowed_gpus), x.max_running, smi_query(x.smi_cmd), x.reserve_seconds)
    wk.start()
    q.log("worker start pid %d allowed_gpus %s max_running %d" % (os.getpid(), wk.allowed, wk.max_running))
    try:
        while True:
            r = wk.tick()
            if x.exit_when_idle and wk.idle():
                q.log("worker idle, exiting")
                break
            if r["stopped"] and r["running"] == 0 and not x.exit_when_idle:
                q.log("queue STOPPED and nothing running; worker exits (fix, then `jobq.py resume`)")
                break
            time.sleep(x.poll)
    finally:
        q.write_status()
        wk.release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
