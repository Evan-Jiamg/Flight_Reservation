"""Local tests for verify_pipeline.py: a clean synthetic run passes; every seeded defect is caught.

Run: python -m pytest -q test_verify_pipeline.py   (or: python test_verify_pipeline.py)
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_pipeline as V  # noqa: E402

SHA_A2 = "a" * 64
SHA_A0 = "0" * 64
SPLITS = {"folds": [{"fold": 0, "train": ["tr1", "tr2"], "validation": ["va1"], "test": ["te1"],
                     "train_all": ["tr1", "tr2", "tr3"], "validation_all": ["va1"], "test_all": ["te1"],
                     "forbidden_for_training": ["te1", "va1"]},
                    {"fold": 1, "train": ["va1"], "validation": ["tr1"], "test": ["te1"],
                     "train_all": ["va1"], "validation_all": ["tr1"], "test_all": ["te1"],
                     "forbidden_for_training": ["tr1", "te1"]}]}
META_A2 = {"arm": "a2", "system_prompt_sha256": SHA_A2, "planner_budget": 32104, "speaker_budget": 32504,
           "act_prior": "nostopclobber", "gate": {"arm": "a2", "fold": 0, "group": 1}}
META_A0 = {"arm": "a0", "system_prompt_sha256": SHA_A0, "planner_budget": 32104, "speaker_budget": 32504,
           "act_prior": "off: original read_plan (override on, length clamp)"}


def a2_prompt(t, gs):
    return ("WHO THEY ARE\n- style: concise\n- gives up: moderately quickly\n\nWHAT THEY CAME FOR\n- topic: x"
            "\n\nWHAT HAS ACTUALLY HAPPENED (counted, not judged)\n- turns so far: %d"
            "\n\n%s\n- status: %s\n- still unmet: %s\n\nTHEY HAVE SENT %d messages"
            "\n\nTHE CONVERSATION SO FAR\nUSER: hi\nASSISTANT: hello" % (t - 1, V.GOAL_HEAD, gs["status"],
                                                                      V.unmet_text(gs), t - 1))


def a2_step(t, status="PARTIAL", unmet=("license info",), end=False, gen=False):
    gs = {"status": "NOT ASSESSED", "unmet": []} if t == 1 else {
        "status": status, "unmet": list(unmet), "parse_ok": status != "UNKNOWN", "raw": '{"status": "%s"}' % status,
        "prompt_tokens": 1200, "dropped_exchanges": 0}
    s = {"t": t, "p_stop": None, "planner_prompt": a2_prompt(t, gs),
         "planner_fit": {"original_tokens": 5000, "removed_tokens": 0, "compacted": False,
                         "prompt_tokens": 5000, "budget": 32104},
         "planner_raw": "{...}", "planner_hit_max_new": False,
         "planner_diag": {"end_session_raw": end, "end_session_valid": True, "length_from_planner": 18,
                          "length_clamped": False, "complete_act_without_end": False},
         "planner_unparsed": False, "ended_planner": end, "move": "Reveal", "act": "request",
         "stop_rule": "none", "goal_status": gs, "ledger_before": {"turns": t - 1, "gain_trace": []}}
    if gen:
        s["planner_gen"] = {"prompt_ids": list(range(5000)), "gen_ids": [1, 2, 3], "temperature": 1.0,
                            "top_p": 1.0, "seed": 7}
    if end:
        s.update(planner_stop=True, user="", decision="planner_stop", emitted=False, agent=None)
    else:
        s.update(user="user message %d" % t, ended_speaker=False,
                 block="- move: Reveal\n- act: request\n- pending: %s\n- length: about 18 words" % V.unmet_text(gs),
                 guard_reasons=None, guard_extra=0, no_survivor=False, selected_index=0, n_candidates=4,
                 speaker_fit={"original_tokens": 3000, "dropped_exchanges": 0, "compacted": False},
                 decision="continue", emitted=True, agent="assistant reply %d" % t)
    return s


def a2_episode(cid="va1", seed=0, gen=False):
    tr = [a2_step(1, gen=gen), a2_step(2, "NOT", ("images", "metadata"), gen=gen), a2_step(3, "UNKNOWN", (), gen=gen),
          a2_step(4, "SATISFIED", (), end=True, gen=gen)]
    return {"conversation_id": cid, "record_id": "r" + cid, "seed": seed, "arm": "a2", "replicate": 0,
            "fold": 0, "group": 1, "emitted_user_turns": 3, "decision_steps": 4, "end_kind": "planner_stop",
            "turns": 3, "trace": tr}


def a0_step(t):
    return {"t": t, "p_stop": None,
            "planner_prompt": ("WHO THEY ARE\n- gives up: quickly (so about 3 unhelpful replies before frustration "
                               "governs)\n\nHOW LONG THEY WRITE\n- band: concise, roughly 8 to 22 words\n\n"
                               "WHAT HAS ACTUALLY HAPPENED (counted, not judged)\n- useful replies: 1\n\n"
                               "WHAT THEY STILL WANT\n- pending: plant images\n\nTHEY HAVE SENT 1"
                               "\n\nTHE CONVERSATION SO FAR\nUSER: hi"),
            "planner_fit": {"original_tokens": 6000, "removed_tokens": 0, "compacted": False},
            "planner_raw": "{}", "planner_diag": {}, "planner_unparsed": False, "ended_planner": False,
            "goal_status": None, "user": "msg %d" % t, "block": "- act: request\n- pending: plant images",
            "speaker_fit": {"original_tokens": 2000, "dropped_exchanges": 0, "compacted": False},
            "decision": "continue", "emitted": True, "agent": "reply"}


def a0_episode(cid="tr1"):
    return {"conversation_id": cid, "seed": 0, "arm": "a0", "fold": -1, "emitted_user_turns": 2,
            "decision_steps": 2, "end_kind": "t_max", "trace": [a0_step(1), a0_step(2)]}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="verify_test_")
        self.splits = self.write("splits.json", SPLITS)

    def write(self, name, obj, jsonl=False):
        p = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            if jsonl:
                for r in obj:
                    f.write(json.dumps(r) + "\n")
            else:
                json.dump(obj, f)
        return p

    def run_v(self, rows, meta, split="validation", fold=0, extra=(), sha=None):
        ep = self.write("ep.jsonl", rows, jsonl=True)
        mp = self.write("meta.json", meta)
        argv = ["--episodes", ep, "--meta", mp, "--splits", self.splits, "--fold", str(fold), "--split", split,
                "--expected-system-sha", sha or (SHA_A2 if meta["arm"] == "a2" else SHA_A0), *extra]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = V.main(argv)
        rep = V.verify([ep], mp, self.splits, fold, split, None, "--training" in extra,
                       sha or (SHA_A2 if meta["arm"] == "a2" else SHA_A0))
        return rc, rep, buf.getvalue()

    def assertCaught(self, rows, meta, check, **kw):
        rc, rep, out = self.run_v(rows, meta, **kw)
        self.assertNotEqual(rc, 0, out)
        self.assertIn(check, rep.checks, out)
        self.assertEqual(rep.status(check), "FAIL", out)
        self.assertIn("PIPELINE VERIFICATION FAILED", out)
        return rep


class CleanRuns(Base):
    def test_clean_a2_passes(self):
        rc, rep, out = self.run_v([a2_episode("va1", 0), a2_episode("va1", 1)], META_A2)
        self.assertEqual(rc, 0, out)
        for name in ("a2.goal_status_in_prompt", "a2.removed_lines_absent", "a2.end_session", "a2.no_stop_override",
                     "a2.no_length_clamp", "a2.pending_line", "a2.judge_outputs", "episode.silent_exit",
                     "trunc.planner", "trunc.speaker", "trunc.judge", "leak.split_membership",
                     "a2.system_prompt_sha"):
            self.assertEqual(rep.status(name), "PASS", name + "\n" + out)
        self.assertIn("PIPELINE VERIFICATION PASSED", out)

    def test_clean_a0_passes(self):
        rc, rep, out = self.run_v([a0_episode("tr1")], META_A0, split="train")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep.status("a0.original_prompt"), "PASS")
        # a0 rows predate planner_hit_max_new -> WARN (not silent), still exit 0 without --strict
        self.assertEqual(rep.status("trunc.planner_max_new"), "WARN")
        rc2, _, _ = self.run_v([a0_episode("tr1")], META_A0, split="train", extra=("--strict",))
        self.assertEqual(rc2, 1)

    def test_training_clean(self):
        rc, _, out = self.run_v([a2_episode("tr1"), a2_episode("tr2")], META_A2, split="train", extra=("--training",))
        self.assertEqual(rc, 0, out)

    def test_status_probs_valid(self):
        ep = a2_episode()
        ep["trace"][1]["goal_status"]["status_probs"] = {"SATISFIED": 0.1, "PARTIAL": 0.2, "NOT": 0.7}
        rc, _, out = self.run_v([ep], META_A2)
        self.assertEqual(rc, 0, out)

    def test_constants_match_planner_prompt_v3(self):
        """The replicated v3 constants must equal planner_prompt_v3 (needs the local sepsim snapshot)."""
        root = os.path.join(HERE, "..", "audit_src")
        if not os.path.isdir(os.path.join(root, "sepsim")):
            self.skipTest("sepsim snapshot not available")
        os.environ["SEPSIM_ACT_PRIOR"] = "nostopclobber"
        sys.path.insert(0, root)
        import planner_prompt_v3 as P3
        self.assertEqual(P3.GOAL_HEAD.strip(), V.GOAL_HEAD)
        self.assertEqual((P3.FIRST_TURN_UNMET, P3.UNKNOWN_UNMET), (V.FIRST_TURN_UNMET, V.UNKNOWN_UNMET))
        for gs in ({"status": "NOT ASSESSED"}, {"status": "UNKNOWN", "unmet": []}, {"status": "NOT", "unmet": ["a", "b"]},
                   {"status": "PARTIAL", "unmet": []}):
            self.assertEqual(P3.unmet_text(gs), V.unmet_text(gs))
        fields = {"move": "Reveal", "act": "request", "length_words": 18}
        blk = P3.speaker_block_v3(fields, {"status": "NOT", "unmet": ["images"]})
        self.assertIn("- pending: images", blk.split("\n"))
        sha = V.recompute_system_sha("a2")
        self.assertTrue(sha and len(sha) == 64)


class SeededDefects(Base):
    def test_over_budget_planner_prompt(self):
        ep = a2_episode()
        ep["trace"][2]["planner_fit"]["prompt_tokens"] = 40000
        self.assertCaught([ep], META_A2, "trunc.planner")

    def test_over_budget_planner_legacy_fields(self):
        ep = a0_episode()
        ep["trace"][0]["planner_fit"] = {"original_tokens": 33000, "removed_tokens": 0, "compacted": False}
        self.assertCaught([ep], META_A0, "trunc.planner", split="train")

    def test_missing_planner_fit(self):
        ep = a2_episode()
        del ep["trace"][1]["planner_fit"]
        self.assertCaught([ep], META_A2, "trunc.planner")

    def test_over_budget_speaker(self):
        ep = a2_episode()
        ep["trace"][1]["speaker_fit"] = {"original_tokens": 40000, "dropped_exchanges": 1, "final_tokens": 33000,
                                         "compacted": True}
        self.assertCaught([ep], META_A2, "trunc.speaker")

    def test_over_budget_judge(self):
        ep = a2_episode()
        ep["trace"][2]["goal_status"]["prompt_tokens"] = 17000
        self.assertCaught([ep], META_A2, "trunc.judge")

    def test_max_new_rate_warns(self):
        ep = a2_episode()
        ep["trace"][0]["planner_hit_max_new"] = True
        rc, rep, out = self.run_v([ep], META_A2)
        self.assertEqual(rep.status("trunc.planner_max_new"), "WARN", out)
        self.assertEqual(rc, 0)

    def test_missing_goal_status(self):
        ep = a2_episode()
        p = ep["trace"][1]["planner_prompt"]
        ep["trace"][1]["planner_prompt"] = p.replace(V.GOAL_HEAD, "")
        self.assertCaught([ep], META_A2, "a2.goal_status_in_prompt")

    def test_wrong_status_line(self):
        ep = a2_episode()
        ep["trace"][1]["planner_prompt"] = ep["trace"][1]["planner_prompt"].replace("- status: NOT\n", "- status: PARTIAL\n")
        self.assertCaught([ep], META_A2, "a2.goal_status_in_prompt")

    def test_leftover_ledger_line(self):
        ep = a2_episode()
        ep["trace"][2]["planner_prompt"] = ep["trace"][2]["planner_prompt"].replace(
            "- turns so far: 2", "- turns so far: 2\n- useful replies: 1")
        self.assertCaught([ep], META_A2, "a2.removed_lines_absent")

    def test_leftover_band(self):
        ep = a2_episode()
        ep["trace"][0]["planner_prompt"] = "HOW LONG THEY WRITE\n- band: concise\n" + ep["trace"][0]["planner_prompt"]
        self.assertCaught([ep], META_A2, "a2.removed_lines_absent")

    def test_override_applied(self):
        ep = a2_episode()
        ep["trace"][1]["planner_diag"]["stop_override"] = True
        self.assertCaught([ep], META_A2, "a2.no_stop_override")

    def test_clamped_length(self):
        ep = a2_episode()
        ep["trace"][2]["planner_diag"]["length_clamped"] = True
        self.assertCaught([ep], META_A2, "a2.no_length_clamp")

    def test_end_session_mismatch(self):
        ep = a2_episode()
        ep["trace"][3]["planner_diag"]["end_session_raw"] = "false"
        self.assertCaught([ep], META_A2, "a2.end_session")

    def test_end_session_not_executed(self):
        ep = a2_episode()
        ep["trace"][1]["planner_diag"]["end_session_raw"] = True
        ep["trace"][1]["ended_planner"] = True          # Planner said stop, but the episode continued
        self.assertCaught([ep], META_A2, "a2.end_session")

    def test_pending_mismatch(self):
        ep = a2_episode()
        ep["trace"][1]["block"] = ep["trace"][1]["block"].replace("- pending: images; metadata", "- pending: plant images")
        self.assertCaught([ep], META_A2, "a2.pending_line")

    def test_silent_exit_with_text(self):
        ep = a2_episode()
        ep["trace"][3]["user"] = "thanks, bye"
        ep["emitted_user_turns"] = 3
        self.assertCaught([ep], META_A2, "episode.silent_exit")

    def test_emitted_count_mismatch(self):
        ep = a2_episode()
        ep["emitted_user_turns"] = 4
        self.assertCaught([ep], META_A2, "episode.emitted_count")

    def test_judge_step1_assessed(self):
        ep = a2_episode()
        ep["trace"][0]["goal_status"] = {"status": "NOT", "unmet": [], "raw": "{}", "prompt_tokens": 10}
        self.assertCaught([ep], META_A2, "a2.judge_outputs")

    def test_judge_raw_missing(self):
        ep = a2_episode()
        del ep["trace"][2]["goal_status"]["raw"]
        self.assertCaught([ep], META_A2, "a2.judge_outputs")

    def test_status_probs_invalid(self):
        ep = a2_episode()
        ep["trace"][1]["goal_status"]["status_probs"] = {"SATISFIED": 0.5, "PARTIAL": 0.2, "NOT": 0.7}
        self.assertCaught([ep], META_A2, "a2.judge_outputs")

    def test_leakage_id_not_in_split(self):
        self.assertCaught([a2_episode("te1")], META_A2, "leak.split_membership")

    def test_leakage_training_forbidden(self):
        # declared as fold-1 train (va1 is fold-1 train) but run meta says fold 0 -> fold mismatch
        self.assertCaught([a2_episode("va1")], META_A2, "leak.meta_fold", split="train", fold=1,
                          extra=("--training",))

    def test_leakage_training_uses_validation(self):
        self.assertCaught([a2_episode("va1")], META_A2, "leak.training_split", extra=("--training",))

    def test_leakage_training_forbidden_list(self):
        splits = copy.deepcopy(SPLITS)
        splits["folds"][0]["forbidden_for_training"].append("tr2")   # e.g. an id moved to validation later
        self.splits = self.write("splits.json", splits)
        self.assertCaught([a2_episode("tr2")], META_A2, "leak.training_forbidden", split="train",
                          extra=("--training",))

    def test_a0_with_v3_lines(self):
        ep = a0_episode()
        ep["trace"][1]["planner_prompt"] = ep["trace"][1]["planner_prompt"].replace(
            "THEY HAVE SENT", V.GOAL_HEAD + "\n- status: NOT\n\nTHEY HAVE SENT")
        self.assertCaught([ep], META_A0, "a0.no_goal_status", split="train")

    def test_a0_missing_original_lines(self):
        ep = a0_episode()
        ep["trace"][0]["planner_prompt"] = ep["trace"][0]["planner_prompt"].replace("WHAT THEY STILL WANT", "")
        self.assertCaught([ep], META_A0, "a0.original_prompt", split="train")

    def test_system_sha_mismatch(self):
        self.assertCaught([a2_episode()], META_A2, "a2.system_prompt_sha", sha="b" * 64)

    def test_meta_act_prior_wrong(self):
        meta = dict(META_A2, act_prior="off")
        self.assertCaught([a2_episode()], meta, "a2.meta_act_prior")

    def test_row_arm_mismatch(self):
        ep = a2_episode()
        ep["arm"] = "a0"
        self.assertCaught([ep], META_A2, "episode.arm")

    def test_bad_jsonl(self):
        ep = self.write("bad.jsonl", [a2_episode()], jsonl=True)
        with open(ep, "a") as f:
            f.write("{not json\n")
        mp = self.write("meta.json", META_A2)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = V.main(["--episodes", ep, "--meta", mp, "--splits", self.splits, "--fold", "0", "--split",
                         "validation", "--expected-system-sha", SHA_A2])
        self.assertEqual(rc, 1)


def wrapped(ep, u, split="train"):
    """train_planner_rl.py rollouts.jsonl row."""
    return {"update": u, "slot": 0, "replicate": 0, "conversation_id": ep["conversation_id"], "split": split,
            "policy_version": u - 1, "policy_sha": "p%d" % u, "episode": ep}


def run_meta_row(kind="start", fold=0, env=None):
    return {"kind": kind, "fold": fold, "config": {"env": dict(env or {k: v for k, v in META_A2.items() if k != "gate"})}}


class RLRuns(Base):
    """Layout of train_planner_rl.py: rollouts.jsonl (wrapped), updates.jsonl, ckpt/uNNNNN, run_meta.jsonl."""

    def make_rl(self, n_updates=3, rollouts=None, updates=None, ckpts=True, meta_rows=None):
        rl = os.path.join(self.dir, "rl")
        os.makedirs(rl, exist_ok=True)
        rows = rollouts if rollouts is not None else [wrapped(a2_episode("tr1", k, gen=True), k)
                                                     for k in range(1, n_updates + 1)]
        self.write("rl/rollouts.jsonl", rows, jsonl=True)
        ups = updates if updates is not None else [
            {"update": k, "train_aggregate": {"reward_mean": 0.3, "components_mean": {
                "goal": 0.5, "over_continue": 0.1, "early_stop": 0.0, "decision_steps": 4.0}}}
            for k in range(1, n_updates + 1)]
        self.write("rl/updates.jsonl", ups, jsonl=True)
        self.write("rl/run_meta.jsonl", meta_rows or [run_meta_row()], jsonl=True)
        if ckpts:
            for u in ups:
                if isinstance(u.get("update"), int):
                    os.makedirs(os.path.join(rl, "ckpt", "u%05d" % u["update"]), exist_ok=True)
        return rl

    def run_rl(self, rl):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = V.main(["--rl-dir", rl, "--splits", self.splits, "--fold", "0", "--split", "train",
                         "--expected-system-sha", SHA_A2])
        rep = V.verify([], os.path.join(rl, "run_meta.jsonl"), self.splits, 0, "train", expected_sha=SHA_A2, rl_dir=rl)
        return rc, rep, buf.getvalue()

    def one(self, mutate):
        ep = a2_episode("tr1", 0, gen=True)
        mutate(ep)
        return [wrapped(ep, 1)]

    def test_clean_rl_passes(self):
        rc, rep, out = self.run_rl(self.make_rl())
        self.assertEqual(rc, 0, out)
        for n in ("rl.planner_gen", "rl.updates_monotone", "rl.checkpoints", "rl.reward_components",
                  "leak.training_train_only", "leak.row_split", "io.meta_consistent", "a2.system_prompt_sha"):
            self.assertEqual(rep.status(n), "PASS", n + "\n" + out)

    def test_flat_rows_and_reward_components_key(self):
        rows = [dict(a2_episode("tr1", 0, gen=True), update=1)]
        ups = [{"update": 1, "reward_components": {"goal": 1.0}, "checkpoint": "ckpt/u00001"}]
        rc, _, out = self.run_rl(self.make_rl(1, rollouts=rows, updates=ups))
        self.assertEqual(rc, 0, out)

    def test_validation_jsonl_as_episodes(self):
        """train_planner_rl validation.jsonl: wrapped episodes + summary rows, checked as fold-0 validation."""
        p = self.write("validation.jsonl", [dict(wrapped(a2_episode("va1"), 1, "validation"), kind="episode"),
                                            {"kind": "summary", "update": 1, "split": "validation"}], jsonl=True)
        mp = self.write("meta.json", META_A2)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = V.main(["--episodes", p, "--meta", mp, "--splits", self.splits, "--fold", "0", "--split",
                         "validation", "--expected-system-sha", SHA_A2])
        self.assertEqual(rc, 0)

    def test_rl_missing_planner_gen(self):
        rows = self.one(lambda ep: ep["trace"][1].pop("planner_gen"))
        rc, rep, out = self.run_rl(self.make_rl(1, rollouts=rows))
        self.assertEqual((rc, rep.status("rl.planner_gen")), (1, "FAIL"), out)

    def test_rl_greedy_rollout(self):
        rows = self.one(lambda ep: ep["trace"][0]["planner_gen"].update(temperature=0.0))
        rc, rep, out = self.run_rl(self.make_rl(1, rollouts=rows))
        self.assertEqual((rc, rep.status("rl.planner_gen")), (1, "FAIL"), out)

    def test_rl_prompt_ids_disagree_with_fit(self):
        rows = self.one(lambda ep: ep["trace"][0]["planner_gen"].update(prompt_ids=[1, 2, 3]))
        rc, rep, out = self.run_rl(self.make_rl(1, rollouts=rows))
        self.assertEqual((rc, rep.status("rl.planner_gen")), (1, "FAIL"), out)

    def test_rl_non_monotone_updates(self):
        ups = [{"update": k, "reward_components": {"goal": 1.0}} for k in (1, 2, 2, 3)]
        rc, rep, out = self.run_rl(self.make_rl(updates=ups))
        self.assertEqual((rc, rep.status("rl.updates_monotone")), (1, "FAIL"), out)

    def test_rl_missing_checkpoint(self):
        rl = self.make_rl()
        os.rmdir(os.path.join(rl, "ckpt", "u00002"))
        rc, rep, out = self.run_rl(rl)
        self.assertEqual((rc, rep.status("rl.checkpoints")), (1, "FAIL"), out)

    def test_rl_missing_reward_components(self):
        ups = [{"update": 1, "reward_components": {"goal": 1.0}}, {"update": 2},
               {"update": 3, "train_aggregate": {"components_mean": {}}}]
        rc, rep, out = self.run_rl(self.make_rl(updates=ups))
        self.assertEqual((rc, rep.status("rl.reward_components")), (1, "FAIL"), out)

    def test_rl_rollout_leak(self):
        rows = [wrapped(a2_episode("va1", 0, gen=True), 1)]
        rc, rep, out = self.run_rl(self.make_rl(1, rollouts=rows))
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep.status("leak.training_forbidden"), "FAIL", out)

    def test_rl_row_split_not_train(self):
        rows = [wrapped(a2_episode("tr1", 0, gen=True), 1, split="validation")]
        rc, rep, out = self.run_rl(self.make_rl(1, rollouts=rows))
        self.assertEqual((rc, rep.status("leak.row_split")), (1, "FAIL"), out)

    def test_rl_meta_changed_on_resume(self):
        env2 = dict({k: v for k, v in META_A2.items() if k != "gate"}, system_prompt_sha256="c" * 64)
        rc, rep, out = self.run_rl(self.make_rl(meta_rows=[run_meta_row(), run_meta_row("resume", env=env2)]))
        self.assertEqual((rc, rep.status("io.meta_consistent")), (1, "FAIL"), out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
