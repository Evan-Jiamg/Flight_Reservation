# -*- coding: utf-8 -*-
"""bench_tf_generate.py without a GPU: conversion to the benchmark generations schema (M2 END rows, samples, flags),
the validator, the gates, and an end-to-end --dry-run (fake env) with resume and provenance."""
import hashlib
import json
import os

import pytest

import bench_tf_generate as B

FOLD = 2


def _rec(cid, n_user, goal_i):
    msgs = []
    for t in range(1, n_user + 1):
        msgs.append({"participant_name": "User", "text": "user %s %d" % (cid, t), "annotations": {}})
        msgs.append({"participant_name": "Agent", "text": "agent %s %d" % (cid, t),
                     "annotations": {"helpful": True, "dataset_quality": "good", "feedback": "x"},
                     "is_final": t == n_user})
    return {"record_id": "rec_" + cid, "conversation_id": cid, "chat_messages": msgs,
            "scenario": {"goal": {"discipline": "Technology", "stage": "focusing", "topic": "g%d" % goal_i},
                         "persona": {"general_info": {"age": str(goal_i)}}}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    test_ids = ["t%02d" % i for i in range(4)]
    train_ids = ["r%02d" % i for i in range(5)]
    n_user = {c: 2 + i % 4 for i, c in enumerate(test_ids)}
    recs = [_rec(c, n_user[c], i) for i, c in enumerate(test_ids)] + [_rec(c, 3, 9) for c in train_ids]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    c_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    bench = tmp_path / "bench"
    (bench / "domains" / "main_dataset_search").mkdir(parents=True)
    man = {"source_sha256": c_sha, "folds": [{"fold": FOLD, "goals": [5, 9, 10], "personas": [2], "session_ids": test_ids,
                                              "n_sessions": 4, "n_user_turns": sum(n_user.values())}]}
    mp = bench / B.BENCH_MANIFEST
    mp.write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(B, "BENCH_MANIFEST_SHA256", hashlib.sha256(mp.read_bytes()).hexdigest())
    monkeypatch.setattr(B, "CORPUS_SHA256", c_sha)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": [{"fold": FOLD, "train": train_ids[:3], "train_all": train_ids,
                                             "validation": [], "test": test_ids[:2], "test_all": test_ids,
                                             "forbidden_for_training": test_ids}]}), encoding="utf-8")
    run = tmp_path / "run"
    for u, psha in ((0, "sha0"), (5, "sha5")):
        d = run / "ckpt" / ("u%05d" % u)
        (d / "adapter").mkdir(parents=True)
        (d / "adapter" / "adapter_model.safetensors").write_bytes(b"w%d" % u)
        (d / "adapter" / "adapter_config.json").write_text("{}")
        (d / "state.json").write_text(json.dumps({"policy_sha": psha}))
        (d / "rl_manifest.json").write_text(json.dumps({
            "arm": "pend", "fold": FOLD, "implicit_profile": 1, "selector": "borda", "fewshot": "fold",
            "planner_backend": "vllm", "planner_path": "/models/q4/", "init_adapter": None,
            "splits_sha256": hashlib.sha256(splits.read_bytes()).hexdigest(),
            "train_scenarios": train_ids[:3], "train_conversations": train_ids, "fewshot_pool": train_ids}))
    (run / "final.json").write_text(json.dumps({"validated": True, "final_update": 5, "policy_sha": "sha5",
                                                "stop_reason": "max_updates"}))
    return {"tmp": tmp_path, "corpus": str(corpus), "bench": str(bench), "splits": str(splits), "run": str(run),
            "test_ids": test_ids, "n_user": n_user, "recs": {r["conversation_id"]: r for r in recs}}


def _argv(w, update=0, out=None, extra=()):
    return ["--final", "--dry-run", "--run-dir", w["run"], "--update", str(update), "--fold", str(FOLD),
            "--splits", w["splits"], "--bench", w["bench"], "--corpus", w["corpus"], "--planner-path", "/models/q4",
            "--workers", "2", "--out", out or str(w["tmp"] / ("gen_u%d.jsonl" % update))] + list(extra)


# ------------------------------------------------------------------ conversion
def _raw(t, greedy, samples, flags=None, prev=False, spk=None, plan_end=False):
    return {"turn_index": t, "greedy": greedy, "samples": samples,
            "samples_speaker_ended": flags if flags is not None else [not s for s in samples],
            "ended_by_prev_decision": prev, "speaker_ended": (not greedy) if spk is None else spk,
            "planner_ends_session": plan_end, "selected_index": 0, "profile": "p"}


def test_conversion_normal_end_and_blank():
    rec = _rec("c1", 4, 1)
    raw = [_raw(1, "hello", ["a", "b", "c", "d"]),
           _raw(2, "bye now", ["x", "", "z"], plan_end=True),        # Planner ends at 2 -> row 3 is END (M2)
           _raw(3, "ignored text", ["q", "r", "s"], prev=True),
           _raw(4, "", ["", "", ""])]                                # blank Ditto pick -> END at row 4
    rows = B.to_benchmark_rows(raw, rec, "iv", max_samples=3)
    assert [r["turn_index"] for r in rows] == [1, 2, 3, 4]
    r1, r2, r3, r4 = rows
    assert r1["greedy"] == "hello" and r1["greedy_end_decision"] is False and r1["samples"] == ["a", "b", "c"]
    assert r1["sample_end_decisions"] == [False, False, False] and r1["is_first_turn"] is True
    assert r2["greedy"] == "bye now" and not r2["greedy_end_decision"] and r2["planner_ends_session"]
    assert r2["sample_end_decisions"] == [False, True, False]        # a blank candidate carries its END flag
    assert r3["greedy"] == "" and r3["greedy_end_decision"] is True and r3["end_source"] == "planner_end_previous_turn"
    assert r3["samples"] == ["", "", ""] and r3["sample_end_decisions"] == [True, True, True]
    assert r4["greedy"] == "" and r4["greedy_end_decision"] and r4["end_source"] == "speaker_blank"
    assert r1["stage"] == "focusing" and r1["discipline"] == "Technology"
    for r in rows:
        assert set(B.REQUIRED_KEYS) <= set(r) and not set(B.RETIRED_KEYS) & set(r)
    B.validate_rows(rows, {rec["record_id"]: 4})


def test_sample_guard_reasons_aligned():
    rec = _rec("c1", 2, 1)
    r1 = _raw(1, "sel", ["a", "b", "c", "d"])
    r1["selected_index"] = 2                        # candidates: a b sel c d -> samples a b c d
    r1["guard_reasons"] = ["near_copy", "", "", "template", "max_new"]
    r2 = _raw(2, "x", ["p", "q", "r"], prev=True)
    r2["guard_reasons"] = ["", "", "", ""]
    rows = B.to_benchmark_rows([r1, r2], rec, "iv")
    assert rows[0]["samples"] == ["a", "b", "c"] and rows[0]["sample_guard_reasons"] == ["near_copy", None, "template"]
    assert rows[0]["selected_guard_reason"] is None
    assert rows[1]["sample_guard_reasons"] == ["end_row"] * 3
    B.validate_rows(rows, {"rec_c1": 2})
    g = B.sample_guard_stats(rows)
    assert g["n_sample_slots"] == 3 and g["n_sample_slots_guard_rejected"] == 2
    assert g["guard_reasons"] == {"near_copy": 1, "template": 1} and g["n_rows_first_nonempty_sample_guard_rejected"] == 1
    bad = [dict(x) for x in rows]
    bad[0]["sample_guard_reasons"] = [None]
    with pytest.raises(ValueError):
        B.validate_rows(bad, {"rec_c1": 2})


def test_length_and_parity_stats():
    rows = [{"greedy": "w " * 250}, {"greedy": "short"}, {"greedy": ""}]
    s = B.length_stats(rows, speaker_tok=lambda t, add_special_tokens=False: {"input_ids": t.split()})
    assert s["n_greedy_over_200_whitespace_tokens"] == 1 and s["n_greedy_over_200_ditto_tokens"] == 1
    assert "not available" in B.length_stats(rows)["ditto_tokens"]
    p = B.decoding_parity()
    assert p["protocol_max_new_tokens"] == {"utterance_generator": 200, "planner": 400}
    assert p["ours_max_new_tokens"]["speaker_ditto"] == 512 and p["ours_max_new_tokens"]["planner"] == 1536


def test_resume_truncated_last_line(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text(json.dumps({"conversation_id": "a"}) + "\n" + '{"conversation_id": "b", "rows": [', encoding="utf-8")
    assert [d["conversation_id"] for d in B.resume_rows(str(p))] == ["a"]
    assert p.read_text(encoding="utf-8") == json.dumps({"conversation_id": "a"}) + "\n"
    p.write_text('{"broken\n' + json.dumps({"conversation_id": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        B.resume_rows(str(p))


def test_foreign_adapters(monkeypatch):
    import types
    import vllm_planner
    monkeypatch.setattr(vllm_planner, "_get", lambda url, timeout=30: {"data": [
        {"id": "planner-base"}, {"id": "p0-abc"}, {"id": "sft_e0-e9629807b9f7"}]})
    remote = types.SimpleNamespace(url="http://x/v1", base_name="planner-base")
    assert B.foreign_adapters(remote, "p0-abc") == ["sft_e0-e9629807b9f7"]


def test_conversion_refuses_too_few_candidates():
    with pytest.raises(ValueError):
        B.to_benchmark_rows([_raw(1, "hi", ["a", "b"])], _rec("c", 1, 1), "iv", max_samples=3)


def test_validator_refusals():
    rec = _rec("c1", 2, 1)
    good = B.to_benchmark_rows([_raw(1, "a", ["x", "y", "z"]), _raw(2, "b", ["x", "y", "z"])], rec, "iv")
    B.validate_rows(good, {"rec_c1": 2})
    bad_cases = []
    r = [dict(x) for x in good]; r[0]["greedy_ended"] = False; bad_cases.append(r)          # retired flag
    r = [dict(x) for x in good]; del r[1]["sample_end_decisions"]; bad_cases.append(r)      # missing key
    r = [dict(x) for x in good]; r[1]["greedy"] = ""; bad_cases.append(r)                  # empty without END
    r = [dict(x) for x in good]; r[0]["greedy_end_decision"] = True; bad_cases.append(r)    # END with text
    r = [dict(x) for x in good][:1]; bad_cases.append(r)                                   # a turn missing
    r = [dict(x) for x in good] + [dict(good[0])]; bad_cases.append(r)                     # duplicate
    r = [dict(x) for x in good]; r[0]["samples"] = ["x", "y"]; r[0]["sample_end_decisions"] = [False, False]
    bad_cases.append(r)                                                                    # not 3 samples
    r = [dict(x) for x in good]; r[0]["samples"] = ["", "y", "z"]; bad_cases.append(r)     # empty sample, no flag
    r = [dict(x) for x in good]; r[1]["is_first_turn"] = True; bad_cases.append(r)
    r = [dict(x) for x in good]; r[0]["greedy"] = "a ⟨redacted:gold⟩"; bad_cases.append(r)
    for rows in bad_cases:
        with pytest.raises(ValueError):
            B.validate_rows(rows, {"rec_c1": 2})


def test_strip_annotations():
    recs = {"c": _rec("c", 3, 1)}
    assert B.strip_agent_annotations(recs) == 3
    assert all(m.get("annotations") == {} for m in recs["c"]["chat_messages"])
    assert [m["text"] for m in recs["c"]["chat_messages"]][:2] == ["user c 1", "agent c 1"]   # text untouched


# ------------------------------------------------------------------ end to end (dry run)
def test_dry_run_end_to_end(world):
    out = str(world["tmp"] / "gen_u0.jsonl")
    prov = B.main(_argv(world, 0, out))
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    expected = {world["recs"][c]["record_id"]: world["n_user"][c] for c in world["test_ids"]}
    assert len(rows) == sum(world["n_user"].values())
    B.validate_rows(rows, expected)
    # exactly the scorer's reading: required fields present, no retired flag, END <=> empty greedy
    for r in rows:
        assert not set(B.RETIRED_KEYS) & set(r)
        assert r["greedy_end_decision"] == (r["greedy"] == "")
        assert r["intent_variant"] == "pend_v17_f2_u0"
    p = json.load(open(out + ".provenance.json", encoding="utf-8"))
    assert p["agent_annotations_stripped"] > 0 and p["summary"]["n_rows"] == len(rows)
    assert p["adapter"]["policy_sha"] == "sha0" and len(p["adapter"]["adapter_dir_sha256"]) == 64
    assert p["planner_gen_adapters"] == ["p0-" + p["adapter"]["adapter_dir_sha256"][:12]]
    assert p["generations_sha256"] == hashlib.sha256(open(out, "rb").read()).hexdigest()
    assert p["settings"]["planner_temperature"] == 0.0 and p["settings"]["end_mapping"] == "M2"
    assert p["cross_check_test_task1"] == {"available": False}
    for k in ("sample_guard_stats", "decoding_parity", "length_stats", "one_draw_note"):
        assert k in p
    assert all(len(r["sample_guard_reasons"]) == len(r["samples"]) for r in rows)
    assert prov["summary"]["n_end_rows"] == sum(1 for r in rows if r["greedy_end_decision"])
    # resumable: a second run regenerates nothing and writes the same file
    before = open(out, "rb").read()
    B.main(_argv(world, 0, out))
    assert open(out, "rb").read() == before
    # ... but refuses other settings on the same --out
    with pytest.raises(SystemExit):
        B.main(_argv(world, 0, out, ["--max-samples", "0"]))


def test_dry_run_deterministic(world):
    a = str(world["tmp"] / "a.jsonl")
    b = str(world["tmp"] / "b.jsonl")
    B.main(_argv(world, 5, a))
    B.main(_argv(world, 5, b))
    assert open(a, "rb").read() == open(b, "rb").read()


def test_gates(world, tmp_path):
    with pytest.raises(SystemExit):                                       # not u0 or the final update
        B.main(_argv(world, 3))
    with pytest.raises(SystemExit):                                       # no --final
        B.main([x for x in _argv(world, 0) if x != "--final"])
    with pytest.raises(SystemExit):                                       # other settings than the run's
        B.main(_argv(world, 0, extra=["--selector", "length"]))
    # an adapter manifest that saw a test id
    mp = os.path.join(world["run"], "ckpt", "u00000", "rl_manifest.json")
    m = json.load(open(mp))
    m["fewshot_pool"] = m["fewshot_pool"] + [world["test_ids"][0]]
    open(mp, "w").write(json.dumps(m))
    with pytest.raises(SystemExit):
        B.main(_argv(world, 0, str(tmp_path / "leak.jsonl")))


def test_corpus_pin(world, monkeypatch):
    monkeypatch.setattr(B, "CORPUS_SHA256", "0" * 64)
    with pytest.raises(SystemExit):
        B.main(_argv(world, 0))


def test_cross_check_with_test_jsonl(world):
    out = str(world["tmp"] / "x.jsonl")
    B.main(_argv(world, 0, out))
    raw = {d["conversation_id"]: d for d in B.read_jsonl(out + ".raw.jsonl")}
    test_rows = [{"kind": "task1", "policy_sha": "sha0", "conversation_id": c,
                  "task1": {"k1_ended": d["k1"]["ended"],
                            "turns": [{"t": r["turn_index"], "ended_planner": r["planner_ends_session"],
                                       "speaker_blank": r["speaker_ended"]} for r in d["rows"]]}}
                 for c, d in raw.items()]
    res = B.compare_with_test(raw, test_rows, "sha0")
    assert res["available"] and res["n_agree"] == res["n_decisions"] > 0 and not res["diffs"]
    c0 = next(iter(raw))
    test_rows[0]["task1"]["k1_ended"] = not test_rows[0]["task1"]["k1_ended"]
    assert B.compare_with_test(raw, test_rows, "sha0")["n_agree"] == res["n_decisions"] - 1
    k1 = [{"record_id": world["recs"][c]["record_id"], "ended": bool(d["k1"]["ended"])} for c, d in raw.items()]
    assert B.compare_with_k1(raw, k1, world["recs"])["n_disagree"] == 0
    k1[0]["ended"] = not k1[0]["ended"]
    assert B.compare_with_k1(raw, k1, world["recs"])["n_disagree"] == 1
    assert c0
