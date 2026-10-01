"""Shell-side tests of start_servers6.sh (sourced with SS6_LIB=1; nvidia-smi / ps / pgrep / pkill / sleep stubbed as
bash functions over files in a fake dir). Needs a POSIX bash (Git Bash on Windows). Run: python -m pytest test_start_servers6.py"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def find_bash():
    """Git Bash first (on Windows `bash` on PATH may be the WSL launcher); a candidate is used only if
    `bash -c 'echo ok'` works -- otherwise the tests are skipped, not failed."""
    for c in (os.environ.get("SS6_BASH"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe",
              shutil.which("bash")):
        if not c or not os.path.exists(c) or "system32" in c.lower():
            continue
        try:
            if subprocess.run([c, "-c", "echo ok"], capture_output=True, text=True, timeout=20).stdout.strip() == "ok":
                return c
        except (OSError, subprocess.SubprocessError):
            continue
    return None


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

STUBS = r'''
SS6_LIB=1 source "$SS6"
H=$FAKE/hold; R=$FAKE/r; Q=$FAKE/q; mkdir -p $H $R $Q
nvidia-smi() {
  case "$*" in
    *"--query-compute-apps"*) cat $FAKE/apps 2>/dev/null ;;
    *"-i "*"memory.free,memory.total"*) g=${2}; echo "$(cat $FAKE/free_$g), 97887" ;;
    *"-i "*"uuid"*) echo "GPU-$2" ;;
  esac
}
ps() { case "$2" in
  user=) cat $FAKE/user_$4 2>/dev/null ;;
  sid=) if [ "$4" = "$$" ]; then cat $FAKE/sid_self; else cat $FAKE/sid_$4 2>/dev/null; fi ;;
esac; }
pgrep() { echo "pgrep $*" >> $FAKE/calls
  case "$*" in
    *"vllm serve"*) [ -s $FAKE/port_pids ] && cat $FAKE/port_pids ;;
    *"-s "*) return 1 ;;
    *) [ -f $FAKE/alive ] ;;
  esac; }
pkill() { echo "pkill $*" >> $FAKE/calls; }
sleep() { :; }
'''


def run(tmp_path, body, files=None):
    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (fake / "hold").mkdir(exist_ok=True)
    for name, text in (files or {}).items():
        p = fake / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, newline="\n")
    script = tmp_path / "t.sh"
    script.write_text(STUBS + body + "\n", newline="\n")
    env = dict(os.environ, FAKE=str(fake).replace("\\", "/"), SS6=os.path.join(HERE, "start_servers6.sh").replace("\\", "/"))
    for k in ("OSS_START_SLACK_GIB", "PLANNER_START_SLACK_GIB", "MIN_START_SLACK_GIB", "MAX_SAME_RACE", "TRAIN_NEED_GIB"):
        env.pop(k, None)
    out = subprocess.run([BASH, str(script).replace("\\", "/")], capture_output=True, text=True, env=env, timeout=60)
    return out.stdout.strip(), out.stderr, fake


def ht(tmp_path, held, free_mib, util, slack, floor):
    out, err, _ = run(tmp_path, "handover_target 0 %s %s %s" % (util, slack, floor),
                      {"hold/status_0": str(held), "free_0": str(free_mib)})
    assert not err.strip(), err
    return out


def test_handover_p1(tmp_path):
    # gpt-oss on the server GPU: 91 held, 1 GiB free -> free >= 0.78 x 95.6 + 3 -> keep 14 for the planner
    assert ht(tmp_path, 91, 1024, 0.78, 3, 0) == "14"
    # then the planner gets everything left (slack "all", floor 0), as start_servers5.sh did
    assert ht(tmp_path, 14, 3482, 0.15, "all", 0) == "0"


def test_handover_p2(tmp_path):
    assert ht(tmp_path, 77, 1024, 0.78, 3, 0) == "0"            # gpt-oss alone: 78 GiB free
    assert ht(tmp_path, 61, 1024, 0.15, 2, 45) == "45"          # planner next to the trainer: the 45 stay held
    assert ht(tmp_path, 61, 0, 0.15, 2, 45) == "45"             # margin eaten by others: still 45 (floor), 16 free
    assert ht(tmp_path, 60, 0, 0.15, 2, 45).startswith("NO")    # only 15 free < 14.3 + 1: do not launch
    assert ht(tmp_path, 44, 17000, 0.15, 2, 45) == "44"         # held below the floor: nothing released (16.6 free)
    assert ht(tmp_path, 44, 5000, 0.15, 2, 45).startswith("NO")  # ... and not enough free: no launch


LOG_STARTUP = ("ValueError: Free memory on device cuda:0 (64.84/94.97 GiB) on startup is less than desired GPU memory "
               "utilization (0.78, 74.08 GiB).\n")
LOG_OOM = "Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"


def classify(tmp_path, log, free_now, apps="", users=None, lf=40000, lp=""):
    files = {"serve.log": log, "free_0": str(free_now), "apps": apps}
    for pid, user in (users or {}).items():
        files["user_%s" % pid] = user
    out, err, _ = run(tmp_path, 'classify_failure $FAKE/serve.log 0 %d "%s"' % (lf, lp), files)
    return out


def test_classify_startup_valueerror_is_a_race(tmp_path):
    assert classify(tmp_path, LOG_STARTUP, 40000).startswith("race")


def test_classify_other_error_is_an_error(tmp_path):
    assert classify(tmp_path, "OSError: model path not found\n", 1000).startswith("error")


def test_classify_oom_needs_evidence(tmp_path):
    assert classify(tmp_path, LOG_OOM, 40000).startswith("error")                    # nobody took memory
    assert classify(tmp_path, LOG_OOM, 30000).startswith("race")                     # free fell by ~10 GiB
    assert classify(tmp_path, LOG_OOM, 40000, "999, GPU-0\n", {999: "other"}).startswith("race")   # new foreign pid
    assert classify(tmp_path, LOG_OOM, 40000, "999, GPU-0\n", {999: "other"}, lp="999").startswith("error")  # was there
    assert classify(tmp_path, LOG_OOM, 40000, "998, GPU-0\n", {998: "mzjiang"}).startswith("error")  # ours
    assert classify(tmp_path, LOG_OOM, 40000, "997, GPU-1\n", {997: "other"}).startswith("error")    # other GPU


def test_three_ambiguous_handovers_stop(tmp_path):
    out, _, _ = run(tmp_path, 'for i in 1 2 3; do strike gpt-oss@g1 && echo "ok $i" || echo "cap $i"; done; '
                              'strike planner@g0 && echo "other ok"')
    assert out.splitlines() == ["ok 1", "ok 2",
                                "STOP: gpt-oss@g1 could not be handed enough memory 3 times on the same placement",
                                "cap 3", "other ok"]


def test_evidenced_races_back_off_and_never_stop(tmp_path):
    out, _, _ = run(tmp_path, 'sleep() { echo "slept $1"; }; for i in 1 2 3 4 5 6; do backoff gpt-oss@g1; done; echo "rc=$?"')
    assert [l for l in out.splitlines() if l.startswith("slept")] == ["slept %d" % d for d in (60, 120, 300, 600, 600, 600)]
    assert out.endswith("rc=0") and "STOP" not in out


def wait_up_out(tmp_path, log, alive):
    files = {"serve.log": log}
    if alive:
        files["alive"] = ""
    out, _, _ = run(tmp_path, 'up() { return 1; }; WAIT_UP_TRIES=6 WAIT_UP_SLEEP=0 wait_up 8029 gpt-oss-120b "port 8029" '
                              '$FAKE/serve.log; echo "rc=$?"', files)
    return out


def test_wait_up_alive_nonfatal_oom_line_is_not_a_failure(tmp_path):
    out = wait_up_out(tmp_path, "flashinfer autotuner: skipping tactic 3: CUDA out of memory\n", alive=True)
    assert "server not up after" in out and "memory" not in out.split("server not up")[0] and out.endswith("rc=1")


def test_wait_up_alive_traceback_alone_is_not_fatal(tmp_path):
    out = wait_up_out(tmp_path, LOG_OOM, alive=True)                       # Traceback + OOM, but no engine failure
    assert "server not up after" in out and not out.startswith("server failed")


def test_wait_up_alive_fatal_memory_error(tmp_path):
    out = wait_up_out(tmp_path, LOG_OOM + "RuntimeError: Engine core initialization failed.\n", alive=True)
    assert out.startswith("server failed (memory):") and out.endswith("rc=1")


def test_wait_up_dead_process(tmp_path):
    out = wait_up_out(tmp_path, LOG_STARTUP, alive=False)
    assert out.startswith("server process died:") and out.endswith("rc=1")


def test_kill_session_scope(tmp_path):
    # our launch = session 777; a port-8029 vllm in session 300 (ours, e.g. its API server) and one that sits in OUR
    # own (pipeline) session 100 -> 777 and 300 are killed, 100 never
    out, err, fake = run(tmp_path, "kill_session 777 8029",
                         {"sid_self": "100", "port_pids": "555\n556\n", "sid_555": "100", "sid_556": "300"})
    calls = (fake / "calls").read_text().splitlines()
    kills = [c for c in calls if c.startswith("pkill")]
    assert "pkill -u mzjiang -s 777" in kills and "pkill -9 -u mzjiang -s 777" in kills
    assert "pkill -u mzjiang -s 300" in kills
    assert not any(" -s 100" in c for c in kills)
    assert all(c.startswith("pkill -u mzjiang ") or c.startswith("pkill -9 -u mzjiang ") for c in kills)


def test_holder_ok_requires_a_fresh_plan(tmp_path):
    out, _, _ = run(tmp_path, 'touch $FAKE/alive; holder_ok && echo A || echo a; touch $H/plan; holder_ok && echo B || echo b; '
                              'touch -d "2 minutes ago" $H/plan; holder_ok && echo C || echo c; rm $FAKE/alive; '
                              'touch $H/plan; holder_ok && echo D || echo d')
    assert out.split() == ["a", "B", "c", "d"]


def test_lost_race_restores_the_target_before_any_sleep(tmp_path):
    """Audit B (final): after an evidenced lost race the placeholder target goes straight back to the pre-hand-over
    value (it re-takes the memory our killed server freed); the back-off sleep comes only before the NEXT launch, while
    the placeholder holds everything again."""
    body = r'''
TG=1; echo 91 > $H/status_0; echo 91 > $H/target_0
write_serve() { :; }; handover_target() { echo 14; }; waitheld_le() { :; }; setsid() { :; }
gpu_mem() { echo "40000 97887"; }; foreign_pids() { :; }; wait_up() { return 1; }
kill_session() { echo "killed target=$(tgt 0)" >> $FAKE/calls; }
classify_failure() { echo "race test"; }
sleep() { echo "sleep $1 target=$(tgt 0)" >> $FAKE/calls; }
start_server oss 0 > /dev/null; echo "rc=$? target=$(tgt 0) next=$NEXT_BACKOFF"
echo "---" >> $FAKE/calls
start_server oss 0 > /dev/null; echo "rc2=$? target=$(tgt 0) next=$NEXT_BACKOFF"
'''
    out, err, fake = run(tmp_path, body)
    lines = out.splitlines()
    assert lines[0] == "rc=75 target=91 next=gpt-oss-120b@g0", (out, err)
    assert lines[1] == "rc2=75 target=91 next=gpt-oss-120b@g0"
    calls = [c for c in (fake / "calls").read_text().splitlines() if not c.startswith("p")]
    assert calls == ["killed target=14", "---", "sleep 60 target=91", "killed target=14"]


# ------------------------------------------------------------------ server ownership (2026-10-01, the 20:03 incident)
CURL = r"""
curl() { u="${@: -1}"; case "$u" in *:8029/*) [ -f $FAKE/answers_8029 ] && cat $FAKE/answers_8029 ;; *:8031/*) [ -f $FAKE/answers_8031 ] && cat $FAKE/answers_8031 ;; esac; [ -n "$(case "$u" in *:8029/*) cat $FAKE/answers_8029 2>/dev/null;; *:8031/*) cat $FAKE/answers_8031 2>/dev/null;; esac)" ]; }
"""


def test_up_requires_our_vllm_process(tmp_path):
    for sub in ("a", "b", "c"):                     # a fresh fake dir per case
        (tmp_path / sub).mkdir()
    body = CURL + 'up 8029 gpt-oss-120b && echo UP || echo DOWN; foreign 8029 && echo FOREIGN || echo NOTFOREIGN'
    # another user's server answers on 8029, no vllm process of ours: not up, foreign
    out, err, _ = run(tmp_path / "a", body, {"answers_8029": '{"data": [{"id": "gpt-oss-120b"}]}'})
    assert out.split() == ["DOWN", "FOREIGN"], out + err
    # our vllm process serves it: up, not foreign
    out, err, _ = run(tmp_path / "b", body, {"answers_8029": '{"data": [{"id": "gpt-oss-120b"}]}', "port_pids": "4242"})
    assert out.split() == ["UP", "NOTFOREIGN"], out + err
    # nothing answers: neither
    out, err, _ = run(tmp_path / "c", body, {})
    assert out.split() == ["DOWN", "NOTFOREIGN"], out + err
