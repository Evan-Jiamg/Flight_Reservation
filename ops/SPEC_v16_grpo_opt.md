# SPEC v16：GRPO 優化（2026-09-28 使用者核准：第 1、2、3、4（下限 0.5）、6、7、8 點；第 5 點不做）

每一項都寫明：改哪裡、確切行為、要記錄什麼、verify 要檢查什麼、要有哪些測試。
實作與稽核一律以本文件為準。本文件沒寫到的行為一律**不變**（原 pend 規格：Implicit Profile、few-shot、Borda、vLLM backend、TIS、stop credit、note mask、reward v4、LLM 控制器、介入機制、重選機制等）。

---

## 項目 1：Task 1 動態補抽（dynamic sampling）

**問題**：Task 1 群組的 G 個樣本若 reward 全相同（std ≤ `min_group_std`），這組沒有梯度，等於白費。u1–u5 每次 8 組裡有 5–8 組是這樣。

**行為**（`train_planner_rl.Trainer.task1_rollouts`）：
1. **基本組**（與現在相同）：用 `rng = Random(seed_of(seed, "task1", u))` 從 `train_all` 抽 `--task1-convs` 段對話，位置也和現在一樣：最後一句 n，以及 n ≥ 3 時的一個較早位置。
   - 每個（對話, 位置）是一組，每組抽 `--task1-G` 個樣本。
   - 這些列標記 `"refill": false`。
2. **目標數** = 基本組中實際抽到的組數（`n_base_groups`）。
3. **有效組**：reward 的母體標準差大於 `acfg["min_group_std"]`，判斷方式和 `advantages_for_groups` 一致。
4. **補抽輪次**：
   - 補抽池 = `train_all` 中這次更新還沒用過的對話；依序從同一個 `rng` 洗牌後取用，洗牌發生在基本組抽完之後。
   - 每一輪先算缺口 `deficit = n_base_groups − 有效組數`。`deficit ≤ 0` 就停止。
   - 否則取接下來的 `min(剩餘池大小, 上限剩餘額度, ceil(deficit / 2))` 段對話（至少 1 段），用同樣的位置規則抽樣，平行執行。
   - 這些列標記 `"refill": true`，重新計算有效組數，再進下一輪。
5. **上限**：補抽對話總數不超過 `--task1-convs`。池空了或達到上限就停。
6. **決定性**：補抽的決定只依賴（rng 順序, 已算出的 reward）。resume 時沿用已寫入的列，所以會得到相同的補抽序列。
7. **aux 範例只取基本組**（`refill == false`），讓 aux 份量固定，不隨補抽變動。
8. **RL 樣本**：基本組加補抽組全部照原本方式變成樣本；無梯度的組照舊跳過並計數。

**記錄**（`task1_train`）：
- 原有欄位：`n`、`acc`、`end_at_final`、`end_at_nonfinal`、`n_not_decisions`、`n_skipped_capped_history`、`groups_skipped_zero_std`。
- 新增：`n_base_groups`、`n_refill_groups`、`n_refill_convs`、`n_informative_groups`、`refill_stop`（`"filled"` / `"pool_empty"` / `"cap"` / `"none_needed"`）。
- `n` / `acc` / `end_at_final` / `end_at_nonfinal` 只用**基本組**計算，才能跨更新比較；另外新增 `acc_all`、`n_all`，涵蓋全部組。`n_not_decisions` 涵蓋全部組。
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
- split 檔讀不到時判為 FAIL，不會默默跳過。

**測試**：
- dry run 中至少有一次更新觸發補抽，而且補抽組的 `refill` 為 true。
- 上限生效。
- resume 後補抽結果相同。
- aux 只取基本組。

---

## 項目 2：Task 1 份量加大

- 新增 `--task1-G`（Task 1 每組樣本數），SPEC 值 **8**，下限 2。Task 2 的 `--G` 維持 4。
- `--task1-convs` 的 SPEC 值 **4 → 8**。
- 兩者和 SPEC 值不同時都要 `--ablation`。
- `task1_sample(..., G=a.task1_G, ...)`。

**verify**：
- `rl.task1_G`：每個 Task 1 列的樣本數 == run_meta 記錄的 `task1_G`。
- 基本組的對話數 == `min(task1_convs, |train_all|)`。

---

## 項目 3：拿掉 GRPO 的組內標準差正規化（Dr. GRPO）

- `rl_algos.ALGO_DEFAULTS["grpo_std_norm"] = False`（新鍵，布林）。
- `group_advantages(rewards, eps, min_std, std_norm)`：
  - `std_norm = False` 時，`A_i = R_i − mean(R)`；
  - `std_norm = True` 時與原本相同（`/ (std + eps)`）。
  - **略過規則不變**：std ≤ `min_std` 就回傳 None。
- `split_group_advantages(rewards, stop_parts, eps, min_std, std_norm)`：
  - `std_norm = False` 時，`A_stop = S − mean(S)`、`A_seq = (R − S) − mean(R − S)`（兩者相加 = `R − mean R`）；
  - 略過規則不變。
- `advantages_for_groups` 把 `cfg["grpo_std_norm"]` 傳進去。所有呼叫點都要傳：Task 2 stop credit、Task 2 無 stop credit、Task 1。
- RLOO、PPO 不受影響。
- run_meta 的 `algo_cfg` 會記錄這個值（原本就會記下整份 acfg）。

**verify**：
- `rl.adv_norm`：run_meta 裡 `algo_cfg.grpo_std_norm is False`。
- `rl.adv_norm` 也要檢查 updates 裡的 `learner_stats`：有樣本的每次更新都必須記錄 `adv_abs_mean`（只看有沒有記錄，不設門檻）。

**測試**：
- 數值：`[0, 1, 1, 0]` → `[-0.5, 0.5, 0.5, -0.5]`。
- split 版兩部分相加等於總和。
- std 為 0 仍然略過。
- `std_norm = True` 的舊行為維持不變。

---

## 項目 4：aux 權重下限（使用者 2026-09-28 決定：0.5）

- 新增 `--stop-sup-floor F`（絕對權重）。
- `aux_weight(u, cfg) = max(F, w · anneal)`：
  - `w = cfg["w_aux"]`；
  - `anneal` 在 D2 未觸發時為 1，觸發後為 `max(F / stop_sup_weight, 1 − (u − start) / stop_sup_anneal)`。退火因子的下限是 F / 初始權重，而不是 0（稽核 P 的 F6）。
  - 也就是 **aux 永遠不會低於 F**；退火結束後，控制器把 `w_aux` 往上調仍然有效（例：w_aux = 2 → 實際 1.0）。
- `--stop-sup-weight 0`（純 GRPO 對照）時，F 必須為 0，否則報錯。
- 當 F 大於「w × anneal」（下限前的值）時，實際權重為 F，並在 hist 記錄 `aux_floor_active: true`。
- **SPEC 值 = 0.5**（使用者 2026-09-28 決定）。考量第 3 點會讓 RL 梯度變小，0.7 可能讓 aux 壓過 RL。其他值需要 `--ablation`。
- 範圍：0 ≤ F ≤ 5。
- LLM 控制器的系統提示中，「it is also annealed to 0 later」改為實作的字句：「once validation Task 1 improves it is annealed towards a fixed floor of %g and never goes below it」，並代入實際的 F。

**記錄**：hist 裡的 `aux_weight` 是實際值，另加 `aux_floor`、`aux_annealed`（下限前的值）、`aux_floor_active`。

**verify**：
- `rl.aux_floor`：
  - `aux_weight ≥ aux_floor`；
  - `aux_weight == max(aux_floor, aux_annealed)`；
  - `aux_weight > 0` 時，`aux_n` 必須**等於**基本組中帶有監督範例的組數（補抽組不可提供範例）。
- `rl.d2_trigger`：只要 `aux_annealed < w_aux`（也就是正在退火），就必須在 D2 觸發之後。

**測試**：
- 衰減到下限就停住。
- F = 0 時與舊行為相同。
- `stop_sup_weight 0` 搭配 F > 0 時報錯。
- 其他 F 值需要 `--ablation`；`--stop-sup-weight 0` 未給 F 時自動視為 0。

---

## 項目 6：w_dist 下限預設固定

- `rl_controllers.LLM4_DEFAULTS["bounds"]["w_dist"] = [1.0, 5.0]`。初始值是 1.0，控制器只能往上調。
- 其他鍵的上下限不變。
- 介入機制保留。

**verify**：
- `rl.w_dist_floor`：run_meta 記錄的控制器下限必須是 1.0（`--ablation` 除外）。
- 每次更新 `cfg_used.w_dist ≥` 下限；使用者核准的介入若改了下限，從介入那次更新起改用介入的下限。

**測試**：控制器提議 0.5 倍時，結果被夾在 1.0。

---

## 項目 7：連續 Task 1 指標（D2 觸發與 checkpoint 選擇改用它）

**定義**：
- 在驗證對話的每個真實回合 t = 2..n（即決策點），取目前政策在該回合 **greedy** 生成的 Planner 輸出，也就是 Task 1 評估所用的同一份 prompt。
- 以 end_session 值之前的前綴為條件，用 learner 做 teacher-forced 計算：`lp_true = log p("true" 值 token)`、`lp_false = log p("false" 值 token)`（`stop_target` 產生兩個版本）。
- `P_end = exp(lp_true) / (exp(lp_true) + exp(lp_false))`。
- 無效決策點（未解析、被 max_new 截斷、end_session 值無效）視為 `P_end = 0`（和 benchmark 一致：未解析就是沒有結束）。
- 決策有效、但找不到值的 token 位置（沒有 stop mask）時，`P_end` 取它的 greedy 決定（1 或 0）。
- 以上兩種都計入 `n_invalid`。
- learner 的計算在 `env.gpu_lock` 之內進行，避免和 Speaker / Planner 同時使用 GPU。
- 指標：
  - `bal_p = ½ · mean_{t=n} P_end + ½ · mean_{2≤t<n} (1 − P_end)`；若沒有任何 t < n 的點，只用前項。
  - 另外記錄：`auc`（最後一句與較早位置的 P_end 排序正確比例，平手算 ½）、`logloss`（對真實標籤的平均負對數機率，P 夾在 [1e-6, 1 − 1e-6]）、`n_points`、`n_final`、`n_invalid`。

**實作**：
- `Task2Env.task1_end_probe(cid, t, user_prompt, real_final)`：
  - 用 `task1_sample(..., G=1, temperature=0.0, top_p=1.0, seed=0)` 取得 greedy 樣本；
  - 有效時回傳 `{"valid": True, "prompt_ids", "prefix_ids", "target_true", "target_false", "greedy_end"}`（呼叫 `stop_target` 兩次）；
  - 否則回傳 `{"valid": False, "greedy_end"}`。
- `TorchLearner.end_prob(x)`：no_grad，用 `token_logprobs(model, prompt + prefix, target, 1.0)` 的總和算上式。
- `FakeLearner.end_prob(x)` = `sigmoid(theta[x["prompt_ids"][0]])`。
- `FakeEnv.task1_end_probe` 相應實作。
- 驗證時的 Task 1：`run_task1(cid, keep_prompts=True)`，照舊算 term_f1 等指標。每個 t ≥ 2 用該回合的 `user_prompt` 呼叫 probe 並算 P_end。寫入 validation.jsonl 前移除 `user_prompt`（避免檔案過大）。
- 每列另存 `"end_probs": [{t, real_final, p_end, valid}]`。
- `task1_stop_metrics` 之外新增 `task1_prob_metrics(rows) → {bal_p, auc, logloss, n_points, n_final, n_invalid}`，放在 `task1_stop.py`，是純 Python。
- summary 的 `task1` 裡加入上述欄位；`task1_base` 也要包含 `bal_p`。

**選擇分數**：
- `selection = w_sel_cov · coverage − w_sel_w1 · W1 + w_sel_task1 · bal_p`。
- summary 記錄 `"selection_task1_metric": "bal_p"` 以及新的 `selection_formula` 字串。
- term_f1 仍然記錄，作為最終報告指標，但不參與選擇。
- 重選（reselect）使用同一個 validate，自動套用。

**D2 觸發**：
- 從「term_f1 > 基準」改為：**連續兩個驗證點**（u > 0）的 `bal_p ≥ base.bal_p + δ`。
- δ = `--t1-trigger-margin`，SPEC 值 **0.10**（約為 20 個點下 bal_p 標準誤的 1.2 倍）。
- 觸發時 `aux_anneal_start = 第二個點的 u`。
- summary 記錄 `d2_trigger: {margin, base_bal_p, streak, triggered_at}`。

**verify**：
- `rl.selection`：依 summary 的 `selection_task1_metric` 重算分數（沒有這個欄位的舊 run 用 term_f1）。
- `rl.task1_prob`：
  - summary 的 `bal_p` 可由該 update 各 task1 列的 `end_probs` 重算出來；
  - 每個 `p_end ∈ [0, 1]`；
  - 決策點數 = Σ(n − 1)。
- `rl.d2_trigger`：
  - 用各 summary 自己的 `task1.bal_p` 對「第一個 summary 的 bal_p + margin」**重新計算** met / streak / 觸發點，和記錄的 `d2` 比對，不採信記錄下來的值；
  - 只能觸發一次，而且位置必須和重算結果相同。

**測試**：
- 指標數值：手算案例的 bal_p、auc、logloss。
- dry run 中 summary 有 `bal_p`，而且可以重算。
- 選擇分數使用 bal_p。
- D2 只在連續兩點達標才觸發。
- 舊格式 summary 的 verify 相容。

---

## 項目 8：跨 fold 合併 Task 1 評估

- 新工具 `task1_pooled.py`。
- 輸入：若干組 `ARM=FILE[,FILE...]`，FILE 是 `task1_v4.py` 產生的 generations `.jsonl`（配合同名 `.k1.jsonl`），每個檔案代表一個 fold 的 test。
- 合併規則：
  - 同一 arm 內各檔的對話 id **不可重複**，因為 fold 的 test 彼此不相交，重複就報錯；
  - 每段對話的回合必須是完整的 1..n，而且有 k1 列。
- 指標：
  - 用 M2 旗標（每列 `greedy_ended`、k1 的 `ended`）算 `term_f1`、`premature_end_rate`、`premature`、`k1_end_rate`；
  - 定義與 `task1_stop.task1_stop_metrics` 完全相同（TP = k1 ended，FP = 真實列上的 END，FN = k1 未 ended）。
- 兩個 arm 之間做配對比較：
  - 要求兩者的對話集合**完全相同**，否則報錯；
  - 以對話為單位做 bootstrap，10000 次、seed 0；
  - 輸出差值、95% CI、`p_b_gt_a` = P(B > A)，以及 `p_b_better`（term_f1 / k1_end_rate 為 B > A，premature 類為 B < A）。
- 輸出可讀文字；給 `--json-out` 時另外寫 JSON；記錄每個輸入檔（generations 和 `.k1.jsonl`）的 sha256。

**測試**：
- 合成檔案計算正確，並與 `task1_stop_metrics` 對照。
- 重複 id 報錯。
- 配對集合不同報錯。
- bootstrap 可重現。

---

## 第 5 點（不做）
理由：能看到的需求數與實際回合數無關（r = −0.36，需求數幾乎都是 7 或 8），而且個人回合數的變化很小（sd 1.1）。對一個無法預測的目標給獎勵，最佳策略就是對每個人都講中間值，等於現狀，只會增加雜訊。

---

## 共通要求
- 所有新 SPEC 值都要進 `parse_args` 的 SPEC gate：`task1_G` 8、`task1_convs` 8、`t1_trigger_margin` 0.10、`stop_sup_floor` 0.5。和 SPEC 值不同就需要 `--ablation`。
- 早先核准的數值也納入 gate（稽核 P 的 F1），不同就需要 `--ablation`：
  - Task 2 的 `G` 4；
  - `behav_mismatch_abort` 0.1；
  - `w_sel_cov`、`w_sel_w1`、`w_sel_task1` 皆為 1.0；
  - `batch` 1；
  - `task1_tol` 0.05；
  - `stop_sup_weight` 1.0 或 0；
  - 非 dry-run 時，`scenarios_per_update` 4、`val_every` 5。
- verify 補強（稽核 P）：
  - v16 run 的選擇指標必須是 `bal_p`；
  - reselect.jsonl 的 summary 也要重算選擇分數與 bal_p；
  - 重算補抽是否該發生、停止原因是否一致；
  - 記錄的 aux_floor 必須等於 run 的 `--stop-sup-floor`；
  - 驗證探測的 greedy 前綴必須由該 update 的 policy adapter 產生（vLLM 的正式 run）。
- 實驗腳本（稽核 Q）：
  - **prelim**：只用 GPU0，佔用中就拒絕；結束或異常退出時一定停掉自己的 server；GPU 讀值不是數字就停止；設時間上限。
  - **step0**：沿對話順序配對回覆；每段對話分別計數 judge 事件，不乾淨的對話不算入 AUC；單段失敗不影響其他段。
  - **formal**：由 `run_v16_launch.sh` 啟動佔位程式，結束時釋放所有 GPU（guard 同樣）；withheld 的驗證不算「沒有改進」；停止規則算不出來就停止。
- `CODE_FILES` 加入 `task1_stop.py`（原本已有）；`task1_pooled.py` 是評估工具，另列。
- provenance：新參數都自動進 run_meta 的 args，resume 時不可改。只有 `RESUME_MAY_CHANGE` 允許更改的例外，而**新參數都不加入**這份清單。
- 既有測試全部通過；新增測試覆蓋每一項。
- 完成後由獨立 sub-agent 對照本文件逐條稽核，確認沒有「要加卻沒加」「只寫文件沒接線」「細節不一致」。
