p = "test_pend.py"
s = open(p, encoding="utf-8").read()
a = s.index("def conv(n, ends):")
b = s.index("def test_parallel_rollouts_equal_serial():")
NEW = '''def conv(n, ends, blanks=(), k1_blank=False):
    return {"k1_speaker_blank": k1_blank,
            "turns": [{"t": t, "real_final": t == n, "ended_planner": t in ends, "speaker_blank": t in blanks,
                       "planner_unparsed": False} for t in range(1, n + 1)]}


def test_task1_stop_metrics():
    # M2: the decision at t is the END flag at row t+1 (K+1 for t = n)
    m = T1.task1_stop_metrics([conv(3, {3}), conv(4, {2}), conv(2, set())])
    # TP=1 (conv1: decision at n -> K+1), FP=1 (conv2: decision at 2 -> END at row 3), FN=2 (conv2, conv3)
    assert m["term_f1"] == pytest.approx(2 / (2 + 1 + 2)) and m["end_mapping"] == "M2"
    assert m["premature"] == pytest.approx(1 / 3) and m["k1_end_rate"] == pytest.approx(1 / 3)
    assert m["premature_end_rate"] == pytest.approx(1 / 9)
    assert m["decision_f1"] == pytest.approx(2 / 5)
    # a Speaker blank is END on its own row; a blank at the probe is a K+1 end
    m2 = T1.task1_stop_metrics([conv(3, set(), blanks={2}), conv(2, set(), k1_blank=True)])
    assert m2["premature_end_rate"] == pytest.approx(1 / 5) and m2["k1_end_rate"] == pytest.approx(1 / 2)
    assert m2["term_f1"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
    # the decision at the last row is NOT an END on that row (M2 shift)
    flags, k1 = T1.bench_flags(conv(3, {3})["turns"])
    assert flags == [False, False, False] and k1
    assert T1.within_tolerance(m, m, 0.0)
    worse = dict(m, premature=m["premature"] + 0.1)
    assert not T1.within_tolerance(worse, m, 0.05) and T1.within_tolerance(worse, m, 0.1 + 1e-9)
    with pytest.raises(ValueError):
        T1.task1_stop_metrics([{"turns": [{"t": 1, "real_final": False, "ended_planner": False}]}])


'''
s = s[:a] + NEW + s[b:]
old = '''        # pend default: reward v3 was used, and validation summaries carry the turn statistics
        assert ua[0]["cfg_used"]["version"] == "v3"'''
assert s.count(old) == 1
s = s.replace(old, '''        # pend default: reward v4 (D1(b)) with the train-only p_h, and validation summaries carry the turn statistics
        assert ua[0]["cfg_used"]["version"] == "v4" and ua[0]["reward_ctx"]["p_h"] and ua[0]["reward_ctx"]["q"]
        assert all(u["train_aggregate"]["shadow_reward_mean"] is not None and u["train_aggregate"]["turn_hist"] for u in ua)''')
old = '''        assert summ and all(s["turn_stats"] and "abs_diff_mean" in s["turn_stats"] for s in summ)'''
assert s.count(old) == 1
s = s.replace(old, '''        assert summ and all(s["turn_stats"] and "abs_diff_mean" in s["turn_stats"] and "turn_w1" in s["turn_stats"] for s in summ)''')
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "test_rl_advantages.py"
s = open(p, encoding="utf-8").read()
old = '''        run_ok(args(sp, o2, controller="llm", updates=2))
        assert len(T.read_jsonl(os.path.join(o2, "llm_controller.jsonl"))) == 2'''
assert s.count(old) == 1
s = s.replace(old, '''        run_ok(args(sp, o2, controller="llm", updates=5))
        # v4 reward -> the factor controller: one decision every 5 updates, factors from the allowed set
        log = T.read_jsonl(os.path.join(o2, "llm_controller.jsonl"))
        assert len(log) == 1 and log[0]["ok"] and log[0]["changed"]
        assert all(v["factor"] == 1.25 for v in log[0]["applied"].values())''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
