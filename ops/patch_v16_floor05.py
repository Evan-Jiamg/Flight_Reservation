"""User 2026-09-28: aux floor = 0.5 (SPEC v16 item 4 value decided)."""
import sys

d = sys.argv[1]


def patch(fn, pairs):
    p = d + "/" + fn
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


patch("sep-sim/train_planner_rl.py", [
    ('''    ap.add_argument("--stop-sup-floor", type=float, default=None,
                    help="v16: the stop supervision never goes below this weight (value pending the user's decision; "
                         "required except with --dry-run, where it defaults to 0)")''',
     '''    ap.add_argument("--stop-sup-floor", type=float, default=0.5,
                    help="v16: the stop supervision never goes below this weight (user 2026-09-28: spec 0.5)")'''),
    ('''    if a.stop_sup_floor is None:
        if not a.dry_run:
            ap.error("--stop-sup-floor is required: its value is pending the user's decision (SPEC v16 item 4)")
        a.stop_sup_floor = 0.0
    if not (0.0 <= a.stop_sup_floor <= 5.0):''',
     '''    if a.stop_sup_floor != 0.5:
        off["stop_sup_floor"] = a.stop_sup_floor
    if not (0.0 <= a.stop_sup_floor <= 5.0):'''),
])
patch("sep-sim/test_v16.py", [
    ('''def test_item4_floor_args():
    base = ["--fold", "0", "--out", "o", "--splits", "s"]
    with pytest.raises(SystemExit):                                             # not dry-run: the value is required
        T.parse_args(base + ["--planner-path", "Qwen3-4B-Instruct-2507"])
    a = T.parse_args(base + ["--dry-run"])
    assert a.stop_sup_floor == 0.0
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--dry-run", "--stop-sup-weight", "0", "--stop-sup-floor", "0.5", "--ablation", "x"])
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--dry-run", "--stop-sup-floor", "6"])
    a = T.parse_args(base + ["--planner-path", "Qwen3-4B-Instruct-2507", "--stop-sup-floor", "0.5"])
    assert a.stop_sup_floor == 0.5 and a.task1_G == 8 and a.task1_convs == 8 and a.t1_trigger_margin == 0.10''',
     '''def test_item4_floor_args():
    base = ["--fold", "0", "--out", "o", "--splits", "s"]
    a = T.parse_args(base + ["--planner-path", "Qwen3-4B-Instruct-2507"])      # user 2026-09-28: spec floor 0.5
    assert a.stop_sup_floor == 0.5 and a.task1_G == 8 and a.task1_convs == 8 and a.t1_trigger_margin == 0.10
    assert T.parse_args(base + ["--dry-run"]).stop_sup_floor == 0.5
    with pytest.raises(SystemExit):                                             # another floor is an ablation
        T.parse_args(base + ["--dry-run", "--stop-sup-floor", "0.7"])
    assert T.parse_args(base + ["--dry-run", "--stop-sup-floor", "0.7", "--ablation", "x"]).stop_sup_floor == 0.7
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--dry-run", "--stop-sup-weight", "0", "--stop-sup-floor", "0.5", "--ablation", "x"])
    assert T.parse_args(base + ["--dry-run", "--stop-sup-weight", "0", "--stop-sup-floor", "0", "--ablation", "x"])
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--dry-run", "--stop-sup-floor", "6", "--ablation", "x"])'''),
    ('''    sp, out = fresh("--stop-sup-floor", "0.5", updates=5)''', '''    sp, out = fresh(updates=5)                                                  # spec floor 0.5'''),
    ('''    sp, out = fresh("--stop-sup-floor", "0.3", updates=4)''', '''    sp, out = fresh("--stop-sup-floor", "0.3", "--ablation", "test-floor", updates=4)'''),
])
patch("ops/SPEC_v16_grpo_opt.md", [
    ('''# SPEC v16：GRPO 優化（2026-09-28 使用者核准：第 1、2、3、6、7、8 點；第 4 點只做機制，數值待定；第 5 點不做）''',
     '''# SPEC v16：GRPO 優化（2026-09-28 使用者核准：第 1、2、3、4（下限 0.5）、6、7、8 點；第 5 點不做）'''),
    ('''## 項目 4：aux 權重下限（只做機制；數值待使用者決定）''', '''## 項目 4：aux 權重下限（使用者 2026-09-28 決定：0.5）'''),
    ('''- SPEC 值 = 待定（`None`）：非 dry-run 時若未明確給 `--stop-sup-floor` 則拒絕執行並提示「數值待使用者決定」；dry-run 未給時視為 0。使用者決定後改為 SPEC 預設值。''',
     '''- SPEC 值 = **0.5**（使用者 2026-09-28 決定；考量第 3 點使 RL 梯度變小，0.7 可能讓 aux 壓過 RL）。其他值需要 `--ablation`。'''),
    ('''- 所有新 SPEC 值都要進 `parse_args` 的 SPEC gate：`task1_G` 8、`task1_convs` 8、`t1_trigger_margin` 0.10、`stop_sup_floor`（待定，必須明確給）；不同就需要 `--ablation`。''',
     '''- 所有新 SPEC 值都要進 `parse_args` 的 SPEC gate：`task1_G` 8、`task1_convs` 8、`t1_trigger_margin` 0.10、`stop_sup_floor` 0.5；不同就需要 `--ablation`。'''),
])
print("patched")
