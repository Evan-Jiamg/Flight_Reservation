"""Old tests updated to the v16 spec (behaviour changes that were approved), old behaviour still checked where it
remains available."""
import sys

d = sys.argv[1]


def patch(fn, pairs):
    p = d + "/" + fn
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


patch("test_rl_advantages.py", [
    ('''def test_group_advantages():
    a = RA.group_advantages([1.0, 0.0, 0.0, 1.0], eps=0.0)
    assert [round(x, 12) for x in a] == [1.0, -1.0, -1.0, 1.0]            # mean .5, population std .5
    a = RA.group_advantages([3.0, 1.0], eps=1e-6)
    assert close(a[0], 1.0 / (1.0 + 1e-6)) and close(sum(a), 0.0)''',
     '''def test_group_advantages():
    # v16 default (Dr. GRPO): A = R - mean(R)
    a = RA.group_advantages([1.0, 0.0, 0.0, 1.0], eps=0.0)
    assert [round(x, 12) for x in a] == [0.5, -0.5, -0.5, 0.5]
    assert RA.ALGO_DEFAULTS["grpo_std_norm"] is False
    # the old normalisation stays available with std_norm=True
    a = RA.group_advantages([1.0, 0.0, 0.0, 1.0], eps=0.0, std_norm=True)
    assert [round(x, 12) for x in a] == [1.0, -1.0, -1.0, 1.0]            # mean .5, population std .5
    a = RA.group_advantages([3.0, 1.0], eps=1e-6, std_norm=True)
    assert close(a[0], 1.0 / (1.0 + 1e-6)) and close(sum(a), 0.0)'''),
    ('''    # selection = w_sel_cov*coverage - w_sel_w1*W1 + w_sel_task1*term_f1, recomputed from the logged parts
    for v in summ:
        if v["selection_score"] is not None:
            t1 = v["task1"]["term_f1"] if v["task1"] else 0.0''',
     '''    # selection = w_sel_cov*coverage - w_sel_w1*W1 + w_sel_task1*bal_p (v16), recomputed from the logged parts
    for v in summ:
        assert v["selection_task1_metric"] == "bal_p"
        if v["selection_score"] is not None:
            t1 = v["task1"]["bal_p"] if v["task1"] else 0.0'''),
])
patch("test_pend.py", [
    ('''            assert r["real_final"] == (r["t"] == r["n_real"]) and len(r["samples"]) == 4''',
     '''            assert r["real_final"] == (r["t"] == r["n_real"]) and len(r["samples"]) == 8      # v16: --task1-G 8'''),
])
print("patched")
