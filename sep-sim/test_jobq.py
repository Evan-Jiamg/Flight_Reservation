"""Local tests for jobq.py: fake nvidia-smi, fake commands (python -c ...), real subprocesses.

Run: python -m pytest -q test_jobq.py   (or: python test_jobq.py)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["JOBQ_HB_SECONDS"] = "0.2"
import jobq  # noqa: E402

PY = '"%s"' % sys.executable
FAKE = os.path.join(tempfile.mkdtemp(prefix="jobq_fake_"), "fake_job.py")
with open(FAKE, "w") as _f:
    _f.write(r'''
import os, sys, time
mode = sys.argv[1]
arg = sys.argv[2] if len(sys.argv) > 2 else ""
if mode == "ok":
    print("hello"); sys.exit(0)
if mode == "fail":
    print("Traceback: ValueError boom", file=sys.stderr); sys.exit(3)
if mode == "oom_once":      # OOM on the first attempt, success afterwards (marker file = arg)
    if os.path.exists(arg):
        sys.exit(0)
    open(arg, "w").close()
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", file=sys.stderr); sys.exit(1)
if mode == "oom":
    print("RuntimeError: CUDA out of memory.", file=sys.stderr); sys.exit(1)
if mode == "quota":
    print("openai.APIStatusError: Error code: 402 - {'error': {'code': 'insufficient_quota'}}", file=sys.stderr)
    sys.exit(1)
if mode == "sleep":
    time.sleep(float(arg)); sys.exit(0)
if mode == "append":         # append the job name to a file (ordering checks)
    time.sleep(0.3)
    with open(arg, "a") as f: f.write(sys.argv[3] + "\n")
    sys.exit(0)
if mode == "env":
    with open(arg, "w") as f: f.write(os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>") + "|" + sys.argv[3])
    sys.exit(0)
''')


def cmd(*args):
    return "%s \"%s\" %s" % (PY, FAKE, " ".join('"%s"' % a for a in args))


class FakeSMI:
    def __init__(self, free):
        self.free = dict(free)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return dict(self.free)


def run_until(wk, cond, timeout=30, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        wk.tick()
        if cond():
            return True
        time.sleep(step)
    wk.tick()
    return cond()


class JobqTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="jobq_test_")
        self.q = jobq.Queue(self.dir)

    def worker(self, gpus=(1,), free=None, **kw):
        kw.setdefault("oom_backoff", (0.2, 1))
        kw.setdefault("fail_backoff", 0.1)
        wk = jobq.Worker(self.q, list(gpus), gpu_query=FakeSMI(free or {0: 80000, 1: 80000, 2: 80000}), **kw)
        wk.start()
        self.addCleanup(wk.release_lock)
        return wk

    def state(self, jid):
        return self.q.load(jid)["state"]

    # ---------------------------------------------------------------- basics
    def test_success_log_heartbeat_status(self):
        self.q.add("a", cmd("sleep", "1.0"), gpu_mib=1000)
        wk = self.worker()
        self.assertTrue(run_until(wk, lambda: self.state("a") == "DONE"))
        j = self.q.load("a")
        self.assertEqual((j["attempts"], j["exit_code"], j["gpu"]), (1, 0, 1))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "hb", "a.hb")))
        log = open(j["log"]).read()
        self.assertIn("[jobq] exit 0", log)
        st = json.load(open(os.path.join(self.dir, "status.json")))
        self.assertEqual(st["counts"]["DONE"], 1)
        self.assertIsNotNone(st["jobs"]["a"]["last_heartbeat_age_s"])
        self.assertEqual(st["jobs"]["a"]["attempts"], 1)

    def test_heartbeat_is_refreshed(self):
        self.q.add("hb", cmd("sleep", "1.5"), gpu_mib=0)
        wk = self.worker()
        wk.tick()
        hb = os.path.join(self.dir, "hb", "hb.hb")
        t0 = time.time()
        while not os.path.exists(hb) and time.time() - t0 < 10:
            time.sleep(0.05)
        m1 = os.path.getmtime(hb)
        time.sleep(0.7)
        self.assertGreater(os.path.getmtime(hb), m1)
        self.assertTrue(run_until(wk, lambda: self.state("hb") == "DONE"))

    def test_env_gpu_pinning(self):
        out = os.path.join(self.dir, "env.txt")
        self.q.add("e", cmd("env", out, "{gpu}/{phys_gpu}"), gpu_mib=100, allowed_gpus=[2])
        wk = self.worker(gpus=(1, 2))
        self.assertTrue(run_until(wk, lambda: self.state("e") == "DONE"))
        self.assertEqual(open(out).read(), "2|0/2")

    # ---------------------------------------------------------------- dependencies
    def test_dependency_ordering(self):
        order = os.path.join(self.dir, "order.txt")
        # added in reverse order on purpose: c after b after a
        self.q.add("c", cmd("append", order, "c"), gpu_mib=10, after=["b"])
        self.q.add("b", cmd("append", order, "b"), gpu_mib=10, after=["a"])
        self.q.add("a", cmd("append", order, "a"), gpu_mib=10)
        wk = self.worker()
        wk.tick()
        self.assertEqual((self.state("a"), self.state("b"), self.state("c")), ("RUNNING", "PENDING", "PENDING"))
        self.assertTrue(run_until(wk, lambda: self.state("c") == "DONE"))
        self.assertEqual(open(order).read().split(), ["a", "b", "c"])

    def test_dependency_on_failed_blocks(self):
        self.q.add("bad", cmd("fail"), gpu_mib=0, retries=0)
        self.q.add("child", cmd("ok"), gpu_mib=0, after=["bad"])
        wk = self.worker()
        self.assertTrue(run_until(wk, lambda: self.state("bad") == "FAILED"))
        for _ in range(3):
            wk.tick()
        self.assertEqual(self.state("child"), "PENDING")
        self.assertTrue(wk.idle())
        self.assertEqual(self.q.summary()["jobs"]["child"]["blocked_by"], ["bad"])

    # ---------------------------------------------------------------- failures
    def test_oom_retry_then_done(self):
        marker = os.path.join(self.dir, "oom_marker")
        self.q.add("o", cmd("oom_once", marker), gpu_mib=10, retries=3)
        wk = self.worker()
        self.assertTrue(run_until(wk, lambda: self.q.load("o")["failures"] == 1))
        j = self.q.load("o")
        self.assertEqual(j["state"], "PENDING")
        self.assertIn("OOM", j["reason"])
        self.assertGreater(j["not_before"], time.time() - 1)
        self.assertTrue(run_until(wk, lambda: self.state("o") == "DONE"))
        j = self.q.load("o")
        self.assertEqual((j["attempts"], j["failures"]), (2, 1))
        self.assertEqual([h["outcome"] for h in j["history"]], ["OOM", "DONE"])

    def test_oom_exhausts_retries(self):
        self.q.add("o2", cmd("oom"), gpu_mib=10, retries=1)
        wk = self.worker()
        self.assertTrue(run_until(wk, lambda: self.state("o2") == "FAILED"))
        self.assertEqual(self.q.load("o2")["attempts"], 2)

    def test_generic_failure_retries_then_failed(self):
        self.q.add("f", cmd("fail"), gpu_mib=0, retries=2)
        wk = self.worker()
        self.assertTrue(run_until(wk, lambda: self.state("f") == "FAILED"))
        j = self.q.load("f")
        self.assertEqual((j["attempts"], j["failures"], j["exit_code"]), (3, 3, 3))
        self.assertTrue(all(h["outcome"] == "FAIL" for h in j["history"]))

    def test_quota_stops_queue(self):
        self.q.add("qa", cmd("quota"), gpu_mib=0, retries=5)
        self.q.add("later", cmd("ok"), gpu_mib=0, after=["qa"])
        self.q.add("indep", cmd("ok"), gpu_mib=0)
        wk = self.worker(max_running=1)
        self.assertTrue(run_until(wk, lambda: self.state("qa") == "QUOTA"))
        self.assertTrue(self.q.stopped())
        for _ in range(3):
            r = wk.tick()
            self.assertTrue(r["stopped"])
            self.assertEqual(r["launched"], 0)
        self.assertEqual(self.state("indep"), "PENDING")       # nothing new starts
        self.assertEqual(self.q.load("qa")["attempts"], 1)       # no retry on quota
        self.assertTrue(self.q.summary()["stopped"])
        jobq.main(["resume", "--queue", self.dir])
        self.assertFalse(self.q.stopped())
        self.assertEqual(self.state("qa"), "PENDING")

    def test_classify_patterns(self):
        self.assertEqual(jobq.classify(0, "CUDA out of memory"), "DONE")
        self.assertEqual(jobq.classify(1, "torch.OutOfMemoryError: CUDA out of memory"), "OOM")
        self.assertEqual(jobq.classify(1, "openai.RateLimitError: insufficient_quota"), "QUOTA")
        self.assertEqual(jobq.classify(1, "HTTP/1.1 402 Payment Required"), "QUOTA")
        self.assertEqual(jobq.classify(1, "step 402 loss 0.3\nValueError"), "FAIL")

    # ---------------------------------------------------------------- crash safety
    def test_dead_pid_requeued_on_restart(self):
        self.q.add("d", cmd("ok"), gpu_mib=10)
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        j = self.q.load("d")
        j.update(state="RUNNING", pid=dead.pid, pgid=None, pid_start=None, attempts=1, gpu=1,
                 started=time.time(), log=os.path.join(self.dir, "logs", "d.a1.log"))
        self.q.save(j)
        wk = self.worker()                      # start() = lock + recover
        j = self.q.load("d")
        self.assertEqual(j["state"], "PENDING")
        self.assertEqual(j["lost"], 1)
        self.assertIn("dead", j["reason"])
        self.assertTrue(run_until(wk, lambda: self.state("d") == "DONE"))
        self.assertEqual(self.q.load("d")["attempts"], 2)

    def test_worker_restart_adopts_live_job(self):
        self.q.add("live", cmd("sleep", "2"), gpu_mib=10)
        wk1 = self.worker()
        wk1.tick()
        self.assertEqual(self.state("live"), "RUNNING")
        wk1.release_lock()                       # simulated worker crash (wrapper keeps running)
        wk2 = self.worker()
        self.assertEqual(self.state("live"), "RUNNING")      # alive -> not requeued
        self.assertTrue(run_until(wk2, lambda: self.state("live") == "DONE"))
        self.assertEqual(self.q.load("live")["attempts"], 1)

    def test_second_worker_refused(self):
        self.worker()
        with self.assertRaises(SystemExit):
            jobq.Worker(self.q, [1], gpu_query=FakeSMI({1: 1})).start()

    def test_atomic_state_files(self):
        self.q.add("x", cmd("ok"), gpu_mib=0)
        leftovers = [n for n in os.listdir(os.path.join(self.dir, "jobs")) if ".tmp." in n]
        self.assertEqual(leftovers, [])

    # ---------------------------------------------------------------- GPU filtering
    def test_gpu0_never_used_unless_allowed(self):
        self.q.add("g", cmd("ok"), gpu_mib=5000)
        wk = self.worker(gpus=(1,), free={0: 80000, 1: 100})
        for _ in range(3):
            wk.tick()
        self.assertEqual(self.state("g"), "PENDING")        # GPU0 is free but not allowed
        wk.gpu_query.free[1] = 9000
        self.assertTrue(run_until(wk, lambda: self.state("g") == "DONE"))
        self.assertEqual(self.q.load("g")["gpu"], 1)

    def test_gpu0_used_when_listed(self):
        self.q.add("g0", cmd("ok"), gpu_mib=5000)
        wk = self.worker(gpus=(0, 1), free={0: 80000, 1: 100})
        self.assertTrue(run_until(wk, lambda: self.state("g0") == "DONE"))
        self.assertEqual(self.q.load("g0")["gpu"], 0)

    def test_job_allowed_gpus_intersect(self):
        self.q.add("j2", cmd("ok"), gpu_mib=100, allowed_gpus=[2])
        wk = self.worker(gpus=(1,), free={1: 80000, 2: 80000})
        for _ in range(3):
            wk.tick()
        self.assertEqual(self.state("j2"), "PENDING")        # job wants 2, worker only has 1

    def test_reservation_prevents_overcommit(self):
        self.q.add("r1", cmd("sleep", "1.5"), gpu_mib=30000)
        self.q.add("r2", cmd("sleep", "1.5"), gpu_mib=30000)
        wk = self.worker(gpus=(1,), free={1: 40000})       # nvidia-smi does not move (fake)
        wk.tick()
        self.assertEqual((self.state("r1"), self.state("r2")), ("RUNNING", "PENDING"))
        self.assertTrue(run_until(wk, lambda: self.state("r2") == "DONE"))

    def test_max_running(self):
        for i in range(3):
            self.q.add("m%d" % i, cmd("sleep", "1"), gpu_mib=0)
        wk = self.worker(max_running=2)
        wk.tick()
        states = sorted(self.state("m%d" % i) for i in range(3))
        self.assertEqual(states, ["PENDING", "RUNNING", "RUNNING"])
        self.assertTrue(run_until(wk, lambda: all(self.state("m%d" % i) == "DONE" for i in range(3))))

    def test_smi_parse_and_failure(self):
        self.assertEqual(jobq.parse_smi("0, 81000\n1, 1200\n2, 40000\n"), {0: 81000, 1: 1200, 2: 40000})

        def boom():
            raise RuntimeError("nvidia-smi missing")
        self.q.add("s", cmd("ok"), gpu_mib=10)
        wk = jobq.Worker(self.q, [1], gpu_query=boom)
        wk.start()
        self.addCleanup(wk.release_lock)
        wk.tick()
        self.assertEqual(self.state("s"), "PENDING")         # no GPU info -> no GPU launch

    # ---------------------------------------------------------------- CLI round trip
    def test_cli_add_worker_status(self):
        smi = "%s -c \"print('1, 50000'); print('0, 80000')\"" % PY
        jobq.main(["add", "--queue", self.dir, "--id", "c1", "--cmd", cmd("ok"), "--gpu-mib", "1000"])
        jobq.main(["add", "--queue", self.dir, "--id", "c2", "--cmd", cmd("ok"), "--gpu-mib", "1000",
                   "--after", "c1"])
        with self.assertRaises(ValueError):
            jobq.main(["add", "--queue", self.dir, "--id", "c1", "--cmd", "x", "--gpu-mib", "1"])
        rc = subprocess.run([sys.executable, os.path.join(HERE, "jobq.py"), "worker", "--queue", self.dir,
                             "--allowed-gpus", "1", "--poll", "0.2", "--exit-when-idle", "--smi-cmd", smi],
                            timeout=60, env=dict(os.environ, JOBQ_HB_SECONDS="0.2")).returncode
        self.assertEqual(rc, 0)
        self.assertEqual((self.state("c1"), self.state("c2")), ("DONE", "DONE"))
        self.assertEqual(self.q.load("c1")["gpu"], 1)
        out = subprocess.run([sys.executable, os.path.join(HERE, "jobq.py"), "status", "--queue", self.dir],
                             capture_output=True, text=True).stdout
        self.assertIn("DONE=2", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
