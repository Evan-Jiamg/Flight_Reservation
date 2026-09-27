"""SPEC v16 text aligned with the audit fixes (2026-09-28)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:60])
    s = s.replace(old, new)


rep('''- `end_at_final` / `end_at_nonfinal` / `acc` 只用**基本組**計算，才能跨更新比較；另外新增 `acc_all`，涵蓋全部組。
- `rollouts_task1.jsonl` 每列加上 `"refill"` 欄位。

**verify**：
- `rl.task1_refill`：補抽對話 ⊂ `train_all`，且和同一更新的基本對話不重複。
- 補抽對話數 ≤ `task1_convs`。
- aux 例子數 ≤ 基本組數。''',
    '''- `n` / `acc` / `end_at_final` / `end_at_nonfinal` 只用**基本組**計算，才能跨更新比較；另外新增 `acc_all`、`n_all`，涵蓋全部組。`n_not_decisions` 涵蓋全部組。
- 群組紀錄（供 verify 使用）：`base_convs`（抽到的基本對話，包括沒有決策位置的）、`refill_convs`、`groups` = [[對話, t, 是否補抽], ...]。
- `rollouts_task1.jsonl` 每列加上 `"refill"` 欄位。

**verify**（稽核後修正版）：
- 以該更新自己記錄的 `groups` 為準，每個 (對話, t) 取最後一列。這樣中止後重跑留下的舊列不會被算進來；沒有決策位置的對話也不會被當成缺漏。
- `rl.task1_refill`：
  - 補抽對話 ⊂ `train_all`，且和基本對話不重複；
  - 補抽對話數 ≤ `task1_convs`；
  - 每組的 `refill` 旗標和清單一致；
  - `n_base_groups` / `n_refill_groups` 與清單一致。
- `rl.task1_G`：`len(base_convs) == min(task1_convs, |train_all|)`；每組樣本數 == `task1_G`。
- split 檔讀不到時判為 FAIL，不會默默跳過。''')
rep('''- `F > w` 時，實際權重為 F，並在 hist 記錄 `aux_floor_active: true`。''',
    '''- 當 F 大於「w × anneal」（下限前的值）時，實際權重為 F，並在 hist 記錄 `aux_floor_active: true`。''')
rep('''- LLM 控制器的系統提示中，「it is also annealed to 0 later」改為：「after validation Task 1 improves it is annealed towards a fixed floor of %g (never below it)」，並代入實際的 F。''',
    '''- LLM 控制器的系統提示中，「it is also annealed to 0 later」改為實作的字句：「once validation Task 1 improves it is annealed towards a fixed floor of %g and never goes below it」，並代入實際的 F。''')
rep('''**記錄**：hist 裡的 `aux_weight` 是實際值，另加 `aux_floor`、`aux_floor_active`。

**verify**：
- `rl.aux_floor`：每次更新的 `aux_weight ≥ aux_floor − 1e-12`。
- `aux_n > 0`：只要 `aux_weight > 0` 且有 Task 1 位置。''',
    '''**記錄**：hist 裡的 `aux_weight` 是實際值，另加 `aux_floor`、`aux_annealed`（下限前的值）、`aux_floor_active`。

**verify**：
- `rl.aux_floor`：
  - `aux_weight ≥ aux_floor`；
  - `aux_weight == max(aux_floor, aux_annealed)`；
  - `aux_weight > 0` 時，`aux_n` 必須**等於**基本組中帶有監督範例的組數（補抽組不可提供範例）。
- `rl.d2_trigger`：只要 `aux_annealed < w_aux`（也就是正在退火），就必須在 D2 觸發之後。''')
rep('''- 非 dry-run 沒給 F 時報錯。''', '''- 其他 F 值需要 `--ablation`；`--stop-sup-weight 0` 未給 F 時自動視為 0。''')
rep('''**verify**：`rl.w_dist_floor`：每次更新的 `cfg_used.w_dist ≥ 1.0`，僅限 v16 之後的 run；判斷方式是 run_meta 的 controller describe 中 bounds 為 1.0。''',
    '''**verify**：
- `rl.w_dist_floor`：run_meta 記錄的控制器下限必須是 1.0（`--ablation` 除外）。
- 每次更新 `cfg_used.w_dist ≥` 下限；使用者核准的介入若改了下限，從介入那次更新起改用介入的下限。''')
rep('''- 無效決策點（未解析、被 max_new 截斷、end_session 值無效、找不到 stop mask）視為 `P_end = 0`（和 benchmark 一致：未解析就是沒有結束），並計數 `n_invalid`。''',
    '''- 無效決策點（未解析、被 max_new 截斷、end_session 值無效）視為 `P_end = 0`（和 benchmark 一致：未解析就是沒有結束）。
- 決策有效、但找不到值的 token 位置（沒有 stop mask）時，`P_end` 取它的 greedy 決定（1 或 0）。
- 以上兩種都計入 `n_invalid`。
- learner 的計算在 `env.gpu_lock` 之內進行，避免和 Speaker / Planner 同時使用 GPU。''')
rep('''- `rl.d2_trigger`：`aux_anneal_start` 只能出現在連續兩點都達標之後。''',
    '''- `rl.d2_trigger`：
  - 用各 summary 自己的 `task1.bal_p` 對「第一個 summary 的 bal_p + margin」**重新計算** met / streak / 觸發點，和記錄的 `d2` 比對，不採信記錄下來的值；
  - 只能觸發一次，而且位置必須和重算結果相同。''')
rep('''  - 輸出差值、95% CI、P(B > A)。
- 輸出 JSON 與可讀文字；寫入各輸入檔的 sha256。''',
    '''  - 輸出差值、95% CI、`p_b_gt_a` = P(B > A)，以及 `p_b_better`（term_f1 / k1_end_rate 為 B > A，premature 類為 B < A）。
- 輸出可讀文字；給 `--json-out` 時另外寫 JSON；記錄每個輸入檔（generations 和 `.k1.jsonl`）的 sha256。''')
rep('''- `rl.adv_norm` 也要檢查 updates 裡的 `learner_stats`：新增 `adv_abs_mean`，只記錄不設門檻。''',
    '''- `rl.adv_norm` 也要檢查 updates 裡的 `learner_stats`：有樣本的每次更新都必須記錄 `adv_abs_mean`（只看有沒有記錄，不設門檻）。''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
