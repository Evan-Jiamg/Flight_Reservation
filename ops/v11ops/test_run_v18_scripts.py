"""Static / dry tests of the SPEC v18 run scripts (no server, no GPU): bash syntax (bash -n), LF line endings, and the
guards every launcher must carry -- the pend_v18 snapshot sha check, the label approval marker (training, smoke), the
placeholder hand-over (gpu_holder3.py + start_servers6.sh, HOLD_DIR), the OOM / GPU-budget retry on THIS attempt's log
lines, --resume (+ --allow-code-change only on request), never writing into the benchmark tree, the chain order.
Plus a behavioural check of the approval guard: run_v18_fold.sh's preamble stops without the marker (stubbed paths).
Needs a POSIX bash (Git Bash on Windows). Run: python -m pytest test_run_v18_scripts.py"""
import os
import re
import subprocess

import pytest

from test_start_servers6 import BASH

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ("run_v18_label.sh", "run_v18_reranker.sh", "run_v18_smoke.sh", "run_v18_fold.sh", "run_v18_test.sh",
           "run_v18_bench.sh", "run_v18_chain.sh")
SEP = os.path.join(HERE, "..", "..", "sep-sim")


def text(name):
    return open(os.path.join(HERE, name), "rb").read()


@pytest.mark.parametrize("name", SCRIPTS)
def test_lf_and_syntax(name):
    raw = text(name)
    assert b"\r\n" not in raw, "%s has CRLF line endings" % name
    if BASH is None:
        pytest.skip("no POSIX bash")
    r = subprocess.run([BASH, "-n", os.path.join(HERE, name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sep_sim_shell_scripts_lf():
    for f in ("bench_score_f2_v18.sh",):
        assert b"\r\n" not in open(os.path.join(SEP, f), "rb").read()


@pytest.mark.parametrize("name", [s for s in SCRIPTS if s != "run_v18_chain.sh"])
def test_common_guards(name):
    s = text(name).decode()
    assert "sha256sum -c --quiet local_sha_v18.txt" in s and "code_snapshots/pend_v18" in s
    assert "gpu_holder3" in s and "HOLD_DIR" in s and "(gpu_holder2|gpu_grab)" in s
    assert 'F" = "2"' in s                                           # the v18 trial is fold 2 only
    assert "/tmp2/hchsu/trec2026-usersim-benchmark/" not in re.sub(r"B=/tmp2/hchsu/trec2026-usersim-benchmark\n", "", s) \
        or name == "run_v18_bench.sh"                                 # only the benchmark step names the (read-only) repo


@pytest.mark.parametrize("name", ["run_v18_reranker.sh", "run_v18_smoke.sh", "run_v18_fold.sh", "run_v18_test.sh",
                                  "run_v18_bench.sh"])
def test_gpu_scripts_hand_over_and_retry(name):
    s = text(name).decode()
    assert "start_servers6.sh" in s and "take_train_gpu" in s and "settarget $TG $TN" in s
    if name != "run_v18_smoke.sh":
        assert "tail -c +$((SZ + 1))" in s and "CUDA out of memory" in s and "seq 1 10" in s


def test_fold_launcher():
    s = text("run_v18_fold.sh").decode()
    assert "--spec v18" in s and "--init-adapter $INIT" in s and "pend_f${F}_v17/ckpt/u00000" in s
    assert "$LABELS.APPROVED" in s and "GPU budget not available" in s and 'RES="--resume"' in s
    assert 'ALLOW_CODE_CHANGE:-0}" = "1"' in s and "--allow-code-change" in s
    assert "PIPELINE VERIFICATION PASSED" in s


def test_test_and_bench_launchers():
    t = text("run_v18_test.sh").decode()
    assert "--test-updates $FU" in t and "--v17-run $RUN17" in t and "eval_test_boot.py" in t and "--include-base" not in t
    b = text("run_v18_bench.sh").decode()
    assert "--update $FU" in b and "$RER" in b and "bench_score_f2_v18.sh $FU" in b and "bench_boot_v18.py" in b
    assert "--baseline-decisions $O17/decisions_u$BU.jsonl" in b and "probe_termination.py" in b
    c = text("run_v18_chain.sh").decode()
    assert c.index("run_v18_fold.sh") < c.index("run_v18_test.sh") < c.index("run_v18_bench.sh")


def test_label_and_reranker_launchers():
    s = text("run_v18_label.sh").decode()
    assert "--split $SPLIT" in s and "train_all validation_all" in s and ".APPROVED" in s
    # audit A: gpt-oss only (no Planner vLLM), and never two labellers / a trainer at once
    assert "OSS_ONLY=1 bash $G/start_servers6.sh" in s and 'pgrep -u mzjiang -f "label_acts.py|train_planner_rl.py' in s
    ss = text("start_servers6.sh").decode()
    assert ss.count('"${OSS_ONLY:-0}" = 1') == 3        # exit when up, exit after the start, skip the 8031 foreign check
    r = text("run_v18_reranker.sh").decode()
    assert '"train_all 0 1" "validation_all 0"' in r and "train_reranker.py fit" in r and "--force-unload" in r


def test_fold_refuses_without_approval(tmp_path):
    """The approval guard of run_v18_fold.sh, run on its preamble with stubbed paths: no marker -> STOP; a marker with the
    label sha -> passes the guard."""
    if BASH is None:
        pytest.skip("no POSIX bash")
    s = text("run_v18_fold.sh").decode()
    start = s.index("LSHA=$(sha256sum $LABELS")
    end = s.index("\n", s.index("grep -q \"$LSHA\" $LABELS.APPROVED"))
    guard = s[start:end + 1]
    lab = tmp_path / "labels.jsonl"
    lab.write_text('{"x": 1}\n')
    script = "set -uo pipefail\nLABELS=%s\n%secho GUARD_OK\n" % (str(lab).replace("\\", "/"), guard)
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert "GUARD_OK" not in r.stdout and "not approved" in r.stdout
    import hashlib
    (tmp_path / "labels.jsonl.APPROVED").write_text(hashlib.sha256(lab.read_bytes()).hexdigest() + "\n")
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert "GUARD_OK" in r.stdout, r.stdout + r.stderr


# ------------------------------------------------------------------ server ownership (2026-10-01, the 20:03 incident)
def _label_guard_script(tmp_path, ours, answers):
    s = text("run_v18_label.sh").decode()
    start = s.index("ours_oss() {")
    end = s.index("\n", s.index('echo "STOP: port 8029 is served by a process that is not ours'))
    stub = ('pgrep() { case "$*" in *"vllm serve"*"8029"*) %s;; *) return 1;; esac; }\n'
            'curl() { %s; }\n' % ("return 0" if ours else "return 1",
                                   "echo '{\"data\": [{\"id\": \"gpt-oss-120b\"}]}'" if answers else "return 7"))
    return "set -uo pipefail\n" + stub + s[start:end + 1] + "echo GUARD_PASSED\n"


@pytest.mark.parametrize("ours,answers,passed", [(False, True, False), (True, True, True), (False, False, True)])
def test_label_refuses_a_foreign_8029(tmp_path, ours, answers, passed):
    if BASH is None:
        pytest.skip("no POSIX bash")
    r = subprocess.run([BASH, "-c", _label_guard_script(tmp_path, ours, answers)], capture_output=True, text=True)
    assert ("GUARD_PASSED" in r.stdout) == passed, r.stdout + r.stderr
    if not passed:
        assert "not ours" in r.stdout


def test_scoring_requires_our_judge_process():
    for f in ("bench_score_f2.sh", "bench_score_f2_v18.sh"):
        s = open(os.path.join(SEP, f), encoding="utf-8").read()
        g = s.index('pgrep -u mzjiang -f "vllm serve .*--port 8029"')
        assert g < s.index('if ! curl -s -m 10 $JUDGE_URL/models'), f


def test_start_servers6_up_is_ownership_checked():
    s = text("start_servers6.sh").decode()
    assert 'up() { pgrep -u $ME -f "vllm serve .*--port $1" > /dev/null && curl' in s
    assert "foreign $port && {" in s
