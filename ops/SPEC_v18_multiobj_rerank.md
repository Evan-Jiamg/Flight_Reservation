# SPEC v18：多目標 GRPO ＋ 擬人度 Reranker（草案 rev 3，2026-10-01，待使用者核准）

**這是唯一的設計，只有一個 arm（不做 ablation，使用者 2026-10-01）。** 實作與稽核一律以本文件為準；本文件沒寫到的行為
沿用 v17（`ops/SPEC_v17_sft_pend.md`，含 §13 的實作決定）；凡本文件與 v17 衝突，以本文件為準。
使用者已決定（2026-10-01）：
1. 兩部分**一起上**：（1）Planner 的多目標 GRPO；（2）Selector 的擬人度 reranker（訓練 rollout 與 test 都用）。
2. **不做 ablation**：不另外訓練或評估「只有 reranker」「只有 GRPO」的版本。評估時只和**既有**的 v17 u0、u5（已評分）
   以及 benchmark 既有方法的結果比較。
3. 起點與 KL 參考都是 v17 的 u0（`runs/pend_f2_v17/ckpt/u00000`），**不重做 SFT**。
4. 只跑 fold 2 試驗，全程（實作＋稽核＋訓練＋評估）約一天。

標 **【待核准】** 的數值或規則是本草案提出、尚未經使用者確認的；§17 列出要使用者決定的問題。
rev 2 依獨立稽核（3 個 BLOCKER、SHOULD、NIT）修訂；**BLOCKER 1（coverage）在使用者回覆前採預設：不用 r_cov**。
rev 3 依複核（4 個 SHOULD、2 個 NIT）修訂：advantage 大小改以 **token 加權**校準（§4.3）、尺度量測的後備與中止處理（§4.1）、
turn 1 的程式路徑（§3.1.5、§18）、Task 2 乾淨 episode 的定義（§3.2）。
rev 4 依再次複核（2 個 SHOULD、NIT）修訂：r_turn 凍結也拒絕訓練、拿掉 κ 後備（§4.1、§4.3）；`clean_v18` 延伸到 validation、test 與
verify（§3.2、§11、§18）；turn 1 無法解析時視為無效（§3.1.5）；J 用名目權重（§7.3）；verify 既有的 clean 缺 judge_error 的 bug（§18）。

---

## 0. 架構前提（不可改）

- 與 v17 §0 相同：arm `pend`（E1.6 tree `trees/e1r_cf19400`）；Planner = Qwen3-4B-Instruct-2507 + LoRA r16，是**唯一**被訓練的
  元件；Speaker = Ditto-8B（凍結，**沒有結束 token**，任何 UserLM 專用機制都不可引入）；Implicit Profile、few-shot（fold 的
  train_all）、R0 = gpt-oss-120b（bench_pin client）、T_MAX 10、第 1 則不能結束、profile_note token 不帶序列 advantage（D4）。
- **v18 只改三件事**：Planner 的 reward 與 advantage（§3–§4）、Selector 多一層 reranker（§6）、validation 與選版規則（§7）。
  Planner／Speaker 的 prompt、few-shot、implicit profile、guards、候選數（1 greedy ＋ 3 samples，NSAMP 3）**全部不改**。
- lr、KL、clip、TIS、epochs × minibatches、群組大小、update 上限沿用 v17；advantage 的**大小**校準到 v17 實測值（§4.3），
  讓每步更新幅度與 RL／aux 的梯度比例維持 v17 的水準。**但 reward 的定義、advantage 的組成與 reranker 都同時改變，
  v18 與 v17 的差異無法歸因到任何單一改動。**

### 0.1 Planner 的 JSON（決定 reward 與 token 遮罩，實際順序）

E1.6 ＋ pend 的輸出欄位依序為：`critique`、`profile_note`、`voice`、`gain`、`case_for_leaving`、`case_for_continuing`、
`goal_met`、`still_wanted`、`stop_rule`、**`end_session`**、`current_stage`、`revealed`、`unrevealed`、`affect`、`patience`、`terms`、
**`act_distribution`**（ACT_FULL：prompt 要求**六個 move 各一筆** `{move, act, p, length_words}`，p 加總為 1；六個 move 不含 Other）、
`length_words`、`length_reason`、`next_step`。
- 實際使用的 act 是從 `act_distribution` **抽樣**（verbalized sampling，`PP.sample_act`）；`end_session=true` 時改用 Complete 那一筆；
  `false` 卻抽到 Complete 時，從非 Complete 各筆重新抽（`read_plan_pend`）。
- `end_session` 的值在 act 與長度**之前**：v17 **Task 1** 的「prefix-only」advantage 從來沒有碰到 act 與長度的 token
  （v17 Task 2 的 coverage advantage 則作用在全部 token）。

## 1. 標註與資料（只用 fold 2 的 train 側；全部快取並記 sha）

### 1.1 真人 act 標籤：我方自己的標註器（**不用** benchmark 的任何標註）

- **對象**：`split["train_all"]` 17 段、共 **79 則**真人訊息（每段 3–7 則）。assert 每個 cid ∈ train_all 且 ∉ forbidden。
- **標註器**：本機 gpt-oss-120b（:8029，與 R0 同一個 server），reasoning effort low。
  - **prompt 只用我方 L2 分類**：`sepsim.acts` 的 6 個 move（Disclose／Reveal／Inquire／Navigate／Note／Complete）＋ Other，
    文字取自 Planner prompt 本身的 `acts.coarse_block()`／`acts.fine_block()`，**不得**含 benchmark
    `instruments/prompts/act_codebook_system.txt` 的文字（實作時以 8-gram 重疊檢查，有重疊就拒絕執行）。
  - **輸入**：只有對話文字（真人 1..t、agent 1..t−1），agent turn 的 `annotations` 一律清空（benchmark 協定
    `t1_teacher_forced.md` 的合法輸入）；**不給** t 之後的任何訊息。
  - **輸出**：`{"move", "act"}`，move 以 `acts.normalise` 正規化；每則訊息標 **K = 3 次**（固定 seed 0/1/2，temperature 1.0）。
  - **失敗處理**：completion 上限 2048 token（gpt-oss 的推理會吃掉預算）；回覆空白或無法解析 → 以新 seed 重試，最多 3 次；
    仍失敗的票記為缺票並計數（`n_votes_missing`、`n_retries`）；一則訊息有效票 < 2 → 該則不給 r_act／r_len 的 move 資訊
    （§3.1 的「不適用」），計數。全部失敗率 > 5% → 停止並回報（不靜默降級）。
- **軟標籤**：q₇(m) = 該則有效票中 move m 的比例（7 類，含 Complete、Other）；fine label 只記錄，不進 reward。
- **品質紀錄**：3 票的 Fleiss κ（coarse）、完全一致率、各 move 的比例；寫入 `act_labels_train_f2.meta.json`。
  **κ 只是標註器的自我一致性，不是正確性**；κ < 0.4 時 §3.4 的 w_act 降為 0.5 並記錄 **【待核准】**。
- **快取**：`/tmp2/mzjiang_usersim/grpo_planner/labels_v18/act_labels_train_f2.jsonl`（含 prompt sha、model、effort、seeds、
  corpus sha、splits sha）。**不得寫進 benchmark tree**（benchmark CLAUDE.md §0 紅線 ①）。
- **validation 標籤另檔**：同一標註器對 `validation_all`（4 段、20 則）標註，存 `act_labels_val_f2.jsonl`，**只供 §7 的 validation
  指標**，訓練程式不得讀取（verify 檢查 reward 用到的標籤 cid 全在 train_all）。
- **test 不標註**：test 側的 act 只由 benchmark 的獨立儀器量測（§8）。

### 1.2 不用 benchmark 的標籤與儀器

- benchmark 有 `instruments/act_annotators/labels_*/acts_full__trec.jsonl`（267 列，**全語料**的真人側 act 標籤，含 fold 2 的
  validation 與 test 段）。它是**評估儀器的輸出**，而且含 test 標籤。
- benchmark CLAUDE.md §0–§1 沒有一條明文禁止「拿它來訓練」（三條紅線管的是：樹內檔案會被同步散布、gold 檢查不可取樣、
  主域計分只用本機 gpt-oss-120b）；但使用者的規則（reward 不得用評估儀器、不得用 validation／test 的 gold 標籤）明確排除它，
  而且用它等於拿 Search-Move Mismatch 的量尺來訓練（Goodhart，§9）。
- **規則**：v18 的 reward、標籤、reranker 都不得讀取 `$BENCH/instruments/` 或 `$BENCH/data/` 底下的檔案。
  - Task 2 rollout 仍需要 pinned `bench_pin/ca13b33/tools/r0_client.py` 的 **R0 client**（task agent，屬於環境）。
  - 同一檔的 **Ledger**（讀 `$BENCH/data/req_shards_v1.json`）是 benchmark 的 Task-2 coverage 量測工具：**預設**它只產生
    報告用的 coverage 診斷值，**不進 reward、不進選版**（§3.2、§7.3）；若使用者在 §17 Q8 選擇保留 r_cov，才改為 reward 項。
  - 這裡的 Ledger 與 Planner prompt 用的 `sepsim.stopping` 輪數／gain 紀錄（`facts_v3(ledger)`）是不同的東西；稽核需確認
    benchmark Ledger 的值沒有進入 Planner 或 Speaker 的輸入。
  - verify 以 grep 程式碼與 open() 稽核紀錄檢查上述規則。

### 1.3 長度與輪數的 gold

- 每則真人訊息的字數 L* = whitespace 分詞數（與 `style_select.n_words` 相同）。fold 2 train_all：中位數 26、四分位 17–35、最大 71。
- 每段 train 對話的真人訊息數 H（v16 reward v3 已用過的同一欄位）。

## 2. 起點與 KL 參考 ＝ v17 u0

- learner 啟動時載入 `runs/pend_f2_v17/ckpt/u00000/adapter`（policy sha 須等於 v17 sft.jsonl 選擇列的
  `f67d643796951c6b…`，不符就拒絕），**複製**（不是 symlink／hard link）為 v18 的 `ckpt/u00000`，複製後再核對 sha。
- `"ref"` adapter 也載入同一份 u0（v17 §2 與 B4 的做法，含 ec496b9 的 bit-exact 重載與 `torch.equal` 檢查）。
- 沒有 SFT 階段：`sft_stage`、`sft_examples`、`base_pend_train` 在 v18 都不執行；`--init-adapter` 為必要旗標
  （`meta()` 目前 assert `init_adapter is None`，v18 改為 assert 存在且 sha 相符，§18）。

## 3. Reward（全部是 train 側的 gold 或環境回饋）

### 3.1 Task 1 群組（teacher-forced，train_all）

- **對話**：每次 update 以 `seed_of(seed, "task1", u)` 從 train_all 抽 `--task1-convs` 8 段（同 v17）。
- **決策點**：每段 **t = 1..n**（v17 是 2..n）；只有一則訊息的對話也有 t = 1 這一點（v17 的 `run_conv` 跳過 n < 2，v18 改掉）。
  截斷歷史的點照 v17 跳過並計數。train_all 共 79 點，比 v17 的 t ≥ 2 多約 27%。
- **prompt**：目前政策的 greedy teacher-forced 執行（`task1_prompts`，Speaker 經 §6 的 selector）。
- **取樣**：每點 G₁ = 4 份計畫，T 1、top_p 1（同 v17）。v17 的 `task1_sample` 在 t < 2 會 raise，v18 改為 t = 1 走 §3.1.5 的有效性定義。

#### 3.1.1 結束 r_stop（t ≥ 2）

`r_stop = 1 − (P_end − y)²`，y = 1[t = n]；P_end 由 learner（π_old）在樣本**自己的前綴**上算（同 v17 §3.2）。
Brier 是 proper scoring rule，期望值在 P_end = 真實結束機率時最大（RLCR 的同一理由）。

#### 3.1.2 act r_act

- **類別**：A = {Disclose, Reveal, Inquire, Navigate, Note}（非 Complete、非 Other；prompt 要求的六個 move 不含 Other，
  所以 Other 不進 reward，否則會變成常數懲罰）。
- **計畫的分布 p**（與 `PP.sample_act` 的丟棄規則相同）：對 `act_distribution` 每一筆
  1. 不是 dict → 丟；`(mv, ac) = acts.normalise(move, act)`；`ac == "other"` 而原始 act 不是 "other" → 丟（格式不良）；
  2. p 不能轉成 float → 丟；p ≤ 0 → 不計入；
  3. **同一個 mv 出現多筆 → p 相加**；
  4. 只保留 mv ∈ A，再正規化成 p(m)。
  - A 上的 p 總和為 0（所有質量都在 Complete／Other）→ p 取 A 上的**均勻分布**（無資訊預測，給中間值，不加倍懲罰；
    是否結束由 r_stop 負責）。記錄 `n_act_uniform_fallback`。
- **真人的 q**：q₇ 去掉 Complete 與 Other 的票後，在 A 上正規化。
- **適用的點**：t < n，且 q₇ 的多數票（最多票者；平手依 `acts.COARSE_ORDER`）不是 Complete 或 Other，且有效票 ≥ 2。
  t = n（act 由結束決定）與其他點不給 r_act，分別計數（`n_act_skipped_final`、`_complete`、`_other`、`_votes`）。
- **式子**：`r_act = 1 − ½ Σ_{m∈A} (p(m) − q(m))²` ∈ [0, 1]。
- 多類 Brier 是 proper scoring rule（Gneiting & Raftery 2007）：期望值在 p = E[q | 狀態] 時最大。**注意它對應的是「標註器的
  票分布」，不是真人的真實 act**；標註器的系統性偏誤會原樣傳給 Planner（§9）。

#### 3.1.3 長度 r_len（所有 t）

- **取哪一筆**：m* = q₇ 的多數 move（平手依 `COARSE_ORDER`）；在 `act_distribution` 中取**第一筆** `acts.normalise` 後 move = m* 且
  `length_words` 可轉為正整數者（與 `read_plan_v3` 的配對方式一致），ℓ̂ = round(該值)。
  - m* = Other，或有效票 < 2 → r_len 不適用，計數。
  - 找不到該 move 的筆，或長度無效 → r_len = 0，計數（`n_len_missing`）。
- **式子**：ρ = min(ℓ̂+1, L*+1) ／ max(ℓ̂+1, L*+1)；`r_len = min(1, ρ ／ ρ₀)`，ρ₀ = 2/3 **【待核准】**（1.5 倍以內滿分）。
- **理由**：比值與尺度無關；帶內無梯度，降低「點估計 reward 把長度推向條件中位數、壓縮長度分布」的壓力
  （RL 降低輸出多樣性見 Kirk et al. arXiv 2310.06452）；取真人多數 move 那一筆，長度分量不會因 act 選錯而被懲罰。

#### 3.1.4 格式 r_fmt（所有 t）

- `r_fmt = 1`（有效）或 `0`（無效）。**無效的樣本只拿 r_fmt，其他分量對它「不適用」**（不再把三個分量都設成 0）。
- 需求揭露（requirement reveal）不做：需求條目來自 `data/req_shards_v1.json`，那是 benchmark 需求條目吻合家族的儀器資料。

#### 3.1.5 樣本狀態

| t | 狀態 | 條件 | 處理 |
|---|---|---|---|
| ≥ 2 | 有效 | 解析成功、未撞 max_new、end_session 值有效、找得到 stop mask | 全部適用的分量 |
| ≥ 2 | 剔除 | 有效但找不到 stop mask | 從群組剔除，無梯度，計數（同 v17） |
| 1 | 有效 | 解析成功、未撞 max_new、`act_distribution` 至少一筆可解析、end_session 值有效（true／false，即使會被忽略），且 `planner.stop_mask(gen_ids)` 找得到該值並通過解碼核對 | r_act、r_len、r_fmt（沒有 r_stop） |
| 1 | 剔除 | 上列都成立，但值遮罩找不到或解碼核對不符 | 剔除（值 token 無法遮罩），計數（與 t ≥ 2 的 stop_mask_mismatch 相同處理） |
| 1 | 無效 | 未解析、撞 max_new、`act_distribution` 無法解析、**或 end_session 缺少／值無效** | 只有 r_fmt = 0（與 t ≥ 2 一致） |
| ≥ 2 | 無效 | 未解析、撞 max_new、值無效 | 只有 r_fmt = 0 |

**turn 1 的程式路徑**（現行程式全部假設 t ≥ 2，v18 逐一改，§18 第 4、7 項）：
- `task2_env.task1_sample`（約第 1317 行）在 t < 2 會 raise → 改為 t = 1 也取樣。
- 同函式的 `valid`（約第 1340–1341 行）要求 `not end_session_t1_ignored`；t = 1 的 end 一定被忽略，所以 t = 1 的樣本永遠「無效」。
  v18 另立旗標 `valid_t1` = 解析成功 ∧ 未撞 max_new ∧ `act_distribution` 可解析 ∧ end_session 值有效（不看它是否被忽略），
  t = 1 的狀態以 `valid_t1` 判定；t ≥ 2 的 `valid`（`decision_valid`）不變。
- `rl_masks`（約第 178–180 行）在 t < 2 回傳 `stop_mask = None` → t = 1 改為直接呼叫 `planner.stop_mask(gen_ids)` 取得值遮罩
  （欄位 `value_mask_t1`），**保留**第 181–187 行的解碼核對：遮罩解碼出的文字必須含有解析到的 `end_session_raw` 值，
  否則視為找不到（剔除，計數 `n_t1_value_mismatch`）。t = 1 的值遮罩只用於「排除 advantage」，不產生 stop credit、不產生 aux。
- `train_planner_rl.task1_samples`（約第 1202 行）`assert r["t"] >= 2` → 改為依 t 分流（t = 1 只有 A_plan）。
- `aux_examples`（約第 1249–1256 行）必須跳過 t = 1（aux 只監督真實的結束決策）。
- `score_task1` 對 t = 1 不計算 P_end、不設 r_stop。
- `run_conv` 不再跳過 n < 2（只有 t = 1 的群組）。
- verify 與測試涵蓋以上每一條（§11、§12）。

### 3.2 Task 2 群組（rollout，train）

- 每次 update 4 個 `train` 情境 × G = 4 個 episode，Planner T 1（同 v17）；Speaker 走 §6 的 selector（含 reranker）。
- 分量：
  - **輪數** `r_turn = 1 − |T − min(H, T_MAX)| ／ T_MAX`（T = 該 episode 的使用者輪數，H = 該 train 對話的真人訊息數；reward v3 的 len_err）；
  - **格式** `r_fmt2 = 1 − (unparsed + hit_max_new + no_survivor 的步數) ／ 決策步數`（v17 的格式率，改成獨立分量）。
- **coverage 預設不進 reward**（§1.2、§17 Q8）；每個 episode 仍記錄 Ledger coverage 作為診斷。
- **乾淨 episode（進入群組的條件）**：v17 的 `episode_clean`（`task2_env.py` 約第 200–204 行；`train_planner_rl.py` 約第 1298 行過濾）
  連 Ledger judge 失敗（`judge_empty`／`judge_unparseable`／`judge_error`）都會丟掉 episode；那是因為 v17 的 reward 用 coverage。
  v18 預設不用 coverage，所以改為 **`clean_v18` = R0 回覆沒有被截斷或空白（`r0_len_truncated == 0` 且 `r0_empty == 0`）
  ∧ 沒有被選中的截斷訊息（emitted_capped）∧ 沒有 compacted 的步**；judge 的三個計數照記錄、該 episode 的 coverage 診斷值記為 null，
  不影響是否進群組。若 §17 Q8 選 (b)（保留 r_cov），恢復 v17 的定義。
- **`clean_v18` 也用在 validation 與 test 的 Task 2**（§7、§8）：judge 失敗的 episode 照常計入輪數，coverage 記為 null、coverage 的
  平均只取非 null 的 episode 並報告缺值數；所以 judge 失敗不會迫使重跑。R0 截斷／空白、emitted_capped、compacted 的 episode 照 v17 重跑。
- `episode["clean"]` 保留 v17 的意義（verify 的 `pend.clean_flag` 照舊檢查）；v18 另存 `episode["clean_v18"]`。
- **取代 v17 的 reward v4**（log p_h(T) − log q(T)）：使用者指定以每個情境的真人輪數為目標；整體分布由 §7、§8 的 turn W1 監看。

### 3.3 不適用的分量

- 一個分量在某樣本「不適用」，該樣本的該分量不參與群組平均，advantage 的該分量為 0。
- 群組內某分量只有 < 2 個適用樣本 → 該分量在此群組全部為 0。

### 3.4 權重 **【待核准】**

| 來源 | 分量 | 權重 w | credit 範圍（§4.4） |
|---|---|---|---|
| Task 1 | r_stop | 1.0 | 前綴（end_session 值之前） |
| Task 1 | r_act | 1.0 | 整個計畫 |
| Task 1 | r_len | 0.5 | 整個計畫 |
| Task 1 | r_fmt | 1.0 | 整個計畫 |
| Task 2 | r_turn | 1.0 | 整個 episode |
| Task 2 | r_fmt2 | 1.0 | 整個 episode |

- r_len 權重低：Speaker 只透過 selector 的長度排名部分實現計畫長度，且要降低壓縮長度分布的風險。
- 權重套在**固定尺度正規化後**的分量上（§4.1），所以權重表示「每單位典型變動」的相對重要性。

## 4. Advantage 與 token 遮罩

### 4.1 每個分量的固定尺度（預先登記，在 u0 的一批 rollouts 上量一次）

1. **量測用的批次** = 第 1 次 update 實際拿來做梯度步的 rollouts（由 u0 產生，尚無梯度步）。在**計算 advantage 之前**，
   對每個分量 k 計算群組內置中值 `c_k,i = r_k,i − mean_{group, 適用}(r_k)`（Dr. GRPO：群組內不除以標準差，arXiv 2503.20783）。
2. **尺度** `s_k = mean(|c_k,i|)`，只取**有 spread 的群組**（該群組該分量的 max − min > 1e-6）中的適用樣本。
   只在非零處取平均，所以「很少有 spread」的分量不會被稀釋出很小的尺度、再放大成很大的 advantage。
3. r_fmt、r_fmt2 是 0／1 的稀有事件，**不量測，固定 s_fmt = 0.5**（一個 4 人群組中單一無效樣本的 |c| 平均為 0.375，與 0.5 同量級）。
4. **量不到的分量一律凍結**（擇一：凍結，不做「下限＋飽和計數」）：r_stop、r_act、r_len、r_turn 中，若有 spread 的群組 < 3 個，
   或量到的 s_k < 0.01 → 該分量**整個 run 的權重設為 0**，寫入 `adv_scales.json` 的 `frozen` 並在報告中列出。
   r_turn 的群組門檻例外：Task 2 每次只有 4 個群組，所以 r_turn 是「有 spread 的群組 < 2 個」才凍結。
   **r_stop、r_act 或 r_turn 被凍結時，不開始訓練**（停下來回報使用者）：前兩者是 Task 1 的主要訊號；r_turn 若凍結，Task 2 只剩
   r_fmt2，κ₂ 會把格式項放大到 v17 整個 Task 2 的 τ。只有 r_len 可以被凍結而繼續訓練。
5. **批次指紋**：`adv_scales.json` 記錄 `batch_sha256` = 量測批次的列身分的 sha256（Task 2：每列的
   `(update, slot, replicate, policy_sha, time)`；Task 1：每列的 `(update, conversation_id, t, policy_sha, time)`，依鍵排序後的 canonical JSON），
   以及群組數、每個分量的有 spread 群組數、s_k、κ（§4.3）、凍結清單。verify 以這個指紋找到**完全相同**的那批列並重算。
6. **update 1 中止時**：若 update 1 留下 `ABORTED_u00001.json`（`train_planner_rl.py` 約第 1056–1058、1138–1141 行：resume 會重新產生
   該次的 rollouts），則舊的 `adv_scales.json` 改名為 `adv_scales.aborted_<batch_sha 前 12 碼>.json` 保留，並在新批次上**重新量測**。
   **一旦 `ckpt/u00001` 存在**（第一個梯度步完成），尺度與 κ 永久凍結：之後的 resume 只讀檔、不重算；檔案的 batch_sha 必須能在
   rollouts 紀錄中找到，否則拒絕繼續。
- **為什麼不用逐次 update 的批次 σ**：rev 1 的批次 σ 在稀有 spread 的分量上會給出 |A| ≈ 5–6，而且 1e-3 的下限會讓 r_stop
  這種小變動分量被靜默丟掉，無效樣本也會主導 σ。固定尺度消除這三個問題，也讓不同 update 的 advantage 可以直接比較。

### 4.2 合成與上界

- 正規化分量 `z_k,i = clip(c_k,i ／ s_k, −3, 3)`（凍結的分量 z = 0）。
- **Task 1**：兩個 advantage
  - `A_pre,i = κ₁ · w_stop · z_stop,i`（只給前綴，§4.4）；
  - `A_plan,i = κ₁ · (w_act·z_act,i + w_len·z_len,i + w_fmt·z_fmt,i)`。
- **Task 2**：`A_ep,i = κ₂ · (w_turn·z_turn,i + w_fmt2·z_fmt2,i)`。
- **上界**（取代 rev 2 的「4 倍目標值」截斷）：z 的截斷已讓每個 advantage 有明確上界，不再另外截斷：
  `|A_pre| ≤ 3κ₁·w_stop`、`|A_plan| ≤ 3κ₁·(w_act + w_len + w_fmt)`、`|A_ep| ≤ 3κ₂·(w_turn + w_fmt2)`；
  前綴 token 同時拿到 A_pre 與 A_plan（A_plan 作用在整個計畫減去值 token），所以前綴 token 的上界是兩者之和
  `3κ₁·(w_stop + w_act + w_len + w_fmt)`。每次 update 記錄 z 被截斷的比例。
- 所有分量都沒有 spread 的群組照 Dr. GRPO 跳過並計數。

### 4.3 大小校準到 v17 實測值：token 加權（BLOCKER 3，rev 3 修正）

**損失怎麼正規化**：learner 每個 minibatch 的損失是 `Σ_samples Σ_tokens (−ratio·a_tok + β·kl) ／ n_tok(minibatch)`
（`rl_algos.py` 約第 819、830、850 行），所以真正決定每步更新大小的是**每個生成 token 平均分到的 |a_tok|**，不是每個樣本的 |A|。
v17 的 Task 1 advantage 只落在有效樣本的前綴 token（`train_planner_rl.py` 約第 1225–1234 行），v18 的 A_plan 落在整個計畫，
token 數約多一倍；若照 rev 2 以「每樣本 |A|」校準，Task 1 的實際更新會比 v17 大約一倍。

**統計量**：`τ_src = Σ_{該來源所有樣本} Σ_{生成 token} |a_tok| ／ Σ_{該來源所有樣本} 生成 token 數`，
其中 a_tok 是 `token_advantages` 的輸出（note、遮罩都已套用；值為 0 的 token 也計入分母，因為損失的分母也是全部 token）。

**v17 的實測值**（唯讀重算：用 `code_snapshots/pend_v17` 的 `rl_reward`／`rl_algos`，由 `updates.jsonl` 的 `cfg_used`、`reward_ctx`、
Task 1 的逐樣本 advantage 紀錄與 `rollouts*.jsonl` 的遮罩重建每個 token 的 advantage；每樣本平均 |A| 與 learner 記錄的
`adv_abs_mean_by_source` 逐次完全相符，作為重建正確的檢查。腳本 `ops/r_v18_inspect6.sh`）：

| 量 | u1 | u2 | u3 | u4 | u5 | 全部（token 加總） |
|---|---|---|---|---|---|---|
| τ Task 1 | 0.0168 | 0.0222 | 0.0167 | 0.0248 | 0.0188 | **0.0197**（379,918 token；非零 132,892） |
| τ Task 2 | 0.0651 | 0.0410 | 0.0827 | 0.1194 | 0.0784 | **0.0762**（219,281 token；非零 166,801） |
| 每樣本平均絕對 A，Task 1（參考） | 0.047 | 0.063 | 0.047 | 0.071 | 0.052 | 0.056 |
| 每樣本平均絕對 A，Task 2（參考） | 0.710 | 0.431 | 0.635 | 0.344 | 0.462 | 0.517 |
| `rl_grad_norm` | 0.0069 | 0.0079 | 0.0102 | 0.0124 | 0.0096 | |
| `aux_grad_norm` | 0.0027 | 0.0022 | 0.0025 | 0.0032 | 0.0025 | |
| 樣本數 Task 1／Task 2 | 124／71 | 104／64 | 100／60 | 100／63 | 112／66 | |

（Task 2 的每樣本平均絕對 A 大，主要來自 v17 stop credit 放在少數值 token 上的 adv_stop；token 加權後只有 0.076。）

- **κ 的定義**：在 §4.1 的量測批次上先以 κ = 1 算出 `τ_src(κ=1)`，令 **`κ₁ = 0.0197 ／ τ_Task1(κ=1)`、`κ₂ = 0.0762 ／ τ_Task2(κ=1)`**。
  κ 與 s_k 一起凍結在 `adv_scales.json`。
- **不需要後備**（rev 3 的 `τ_v17／0.75` 後備已拿掉）：§4.1 的拒絕規則保證兩個來源的 τ(κ=1) > 0——r_stop 與 r_act 未凍結 ⇒ Task 1
  有 spread；r_turn 未凍結 ⇒ Task 2 有 spread。若實作時仍遇到 τ(κ=1) = 0，視為程式錯誤，停止。
- **κ₂ 的信心**：r_turn 恰好只有 2 個群組有 spread 時，κ₂ 照算並標記 `kappa2_low_confidence`（報告中列出）；≥ 3 個則不標記。
- **Task 1 與 Task 2 的相對權重**因此與 v17 相同（token 加權約 0.020 對 0.076）；兩個來源照 v17 一起進入同一個 token-sum 損失。
  Dr. GRPO 的 token 加總會讓較長的 Task 2 episode（步數多）權重較大，這點與 v17 相同，照實記錄。
- **梯度裁切**：v17 的總梯度範數約 0.01–0.016，遠低於裁切值 1.0，裁切從未作用；所以 RL 與 aux 的相對強度由 τ 決定。
  v17 的 `rl_grad_norm ／ aux_grad_norm` 為 2.6–3.9（逐次 update 平均）。
- **警示**（每次 update 記錄、不自動調整）：
  - `τ_src` 超出 v17 逐次範圍的 2 倍（Task 1 > 0.050 或 Task 2 > 0.239）；
  - `rl_grad_norm ／ aux_grad_norm` 的 update 平均 > 8（v17 最大值 3.9 的約 2 倍），或 `rl_grad_norm` 平均 > 0.025（v17 最大 0.0124 的 2 倍）。
  有了 token 加權的校準，這些比例在 u1 應落在 v17 的範圍內；超出代表校準或遮罩出了問題，警示才有意義。
- lr 1e-5、clip 0.2、aux 0.5 因此不改；**這不表示 v18 的效果可歸因**（§0）。

### 4.4 哪些 token 拿到 advantage

| 樣本 | token advantage |
|---|---|
| Task 1 有效，t ≥ 2 | `A_pre · prefix_mask · (1−note)` ＋ `A_plan · (1−note) · (1−stop_mask)`；prefix_mask = `prefix_mask_of(stop_mask)`（值 token 之前） |
| Task 1 有效，t = 1 | `A_plan · (1−note) · (1−value_mask)`；value_mask = `planner.stop_mask(gen_ids)`（t = 1 的值不是決策，也不給 advantage） |
| Task 1 無效 | `A_plan · (1−note)`（此時 A_plan 只有 r_fmt 的項） |
| Task 2 每一步 | `A_ep · (1−note)`，**包含** end_session 值 token；`--stop-credit 0`（v17 的 stop credit 關掉） |

- r_stop 只給前綴：P_end 是在前綴上算的，值之後的 token 不影響它；給整個計畫只會加雜訊。
- Task 1 的值 token 不給 advantage：r_stop 已對兩個值取機率，值 token 由輔助監督（aux，權重 0.5，t ≥ 2，同 v17 §3.3）訓練。
- `token_advantages` 改成 `A_pre·prefix_mask + A_plan·plan_mask + A_ep·ep_mask`；assert 各遮罩長度等於生成長度、
  prefix_mask 與 stop_mask 不相交、Task 1 的值 token 上 advantage 為 0。

## 5. learner 更新

- epochs 2 × minibatches 4 ＝ 每次 update 8 個 optimizer step；clip 0.2；TIS cap 2（|Δlogp| > 0.1 中止）；lr 1e-5；
  AdamW、梯度裁切 1.0；KL k3 0.01，參考 = u0；aux 0.5（每個 t ≥ 2 決策點一筆，分到 4 個 minibatch）。
- **update 次數**：最多 `--updates 5`（同 v17），可被 §7.3 的停止規則提早結束。
- **崩潰監看**（v17 §3.5 全部保留，另加）：
  - Task 1：各分量的平均與 spread 群組數、τ_Task1（§4.3 的 token 加權統計）與每樣本 `adv_abs_mean`（分 pre／plan）、z 被截斷的比例、
    計畫 act 分布在 A 上的平均熵、計畫長度的四分位距、`rl_grad_norm ／ aux_grad_norm`、§4.3 的警示；
  - Task 2：τ_Task2、平均輪數、turn_hist、coverage（診斷）、judge 失敗計數（診斷）、`clean_v18` 剔除數與原因、reranker 改變選擇的比例（§6.4）；
  - 長度漂移停止沿用 v17（連續 2 次 mean(T − min(H, 10)) < −1.0）。

## 6. Selector 的擬人度 reranker

### 6.1 資料（只用 train 側）

- **正例**：train_all 的 79 則真人訊息。
- **負例**：在 train_all 每一個 teacher-forced 狀態（cid, t），用 **u0** ＋ 目前的 Borda selector 跑 `run_task1`（Speaker 照常產生
  4 個候選），seeds 0 與 1。
  - **去重**：slot 0 是 Ditto 的 greedy（T 0），兩個 seed 常產生相同文字；同一狀態內文字相同（正規化空白後）的候選只留一個。
  - 去掉空白與 guard 拒絕的候選。
- **成對**：同一狀態的（真人訊息, 我方候選）組成一對；報告**有效樣本數**：獨立的對數、狀態數（≤ 79）、對話數（17，統計上的
  獨立單位是對話）。預期約 400–600 對。
- **不用**：fold 2 的 validation／test 段、其他 fold 的段（與 fold 2 test 共用 goal 或 persona，正是 goal-persona 切分排除它們的原因）、
  benchmark 的任何 generations 或結果檔。

### 6.2 模型

- **特徵** φ(x)：SimCSE（`sup-simcse-bert-base-uncased`，selector 已載入，CPU）句向量經 PCA 降到 32 維；
  加 5 個風格特徵（小寫開頭、句尾有無標點、問號數、驚嘆號數、大寫字元比例）。**不含長度**（Borda 已有）、**不用 TF-IDF**（§9）。
- **成對 logistic regression**（Bradley–Terry）：`P(h ≻ c) = σ(w · (φ(h) − φ(c)))`，無截距、L2。
- **leave-one-conversation-out**（17 折）：**PCA 與特徵標準化都在每一折的訓練部分內擬合**；C 從 {0.01, 0.1, 1} 以 LOCO 成對正確率選出；
  最終模型以選定的 C 在全部 train 資料上重擬（含 PCA）。擬人度分數 `s(x) = w · φ(x)`。
- **為什麼成對**：同一狀態的內容相近，差異主要是寫法；正例只有 79 則，同狀態配對把有限的資料集中在風格上
  （residual EBM 用真／假判別器重排 LM 樣本，Deng et al. arXiv 2004.11714）。
- **校準**：以 LOCO 的 out-of-fold 分數做 Platt scaling，報告 Brier 與 reliability；selector 只用排名，校準只供診斷。
- **產物**：`reranker_v18_f2.json`（PCA 參數、w、C、特徵定義、訓練 cid、資料 sha、LOCO 指標、有效樣本數）。

### 6.3 採用門檻（預先登記）

- **validation 的候選**：用 u0 ＋ Borda 在 `validation_all`（4 段、20 則）跑一次 teacher-forced `run_task1`（seed 0），取得同狀態的候選
  （只評估、不訓練；需要 Planner vLLM ＋ Ditto，約 10 分鐘，已列入 §13）。
- **門檻**：LOCO 成對正確率 ≥ 0.60，且 validation 成對正確率 ≥ 0.55（點估計）。
  - 另報告兩者的對話層級 cluster bootstrap 95% CI（LOCO：17 群；validation：只有 4 群，CI 會非常寬，照實報告）。
- 不過門檻 → **不用 reranker**（selector 退回 Borda），GRPO 照常跑，並在報告中寫明。這是門檻，不是額外的 arm。

### 6.4 與 Borda 的結合：Borda 先篩、reranker 再選 **【待核准，§17 Q4】**

1. 候選池與 v17 相同（guard 通過者；全滅時的後備規則不變）。
2. Borda（長度＋SimCSE 風格）照舊計分；取 Borda 分數最好的前 2 名（第 2 名同分者全部納入）。
3. 在這些候選中選 `s(x)` 最高者；同分依 Borda 分數、再依候選順序。
- 每一步都記錄 `borda_index`（v17 會選的那個）、`rerank_scores`、`rerank_changed`，供 §5 監看與 §8 報告「改變選擇的比例」
  （只描述，不另做 Borda-only 的評估）。
- **作用範圍**：Task 1 prompt 的 teacher-forced 執行、Task 2 訓練 rollout、validation、test、benchmark 生成，**全部**相同。
- **reranker 不進 Planner 的 reward**（§9）。
- 只有 4 個候選：best-of-n 相對原分布的 KL 有 `log n − (n−1)/n ≈ 0.64` nats 的參考值（Beirami et al. arXiv 2401.01879）；
  這裡是在 Borda 前 2 名中選、而且分數並非 reward，所以只當作「偏離幅度有限」的啟發式說明，不是保證。

### 6.5 Benchmark 的取樣通道與 2AFC 的混淆（重要）

- `bench_tf_generate.py` 沿用 sep1st 的對應：`greedy` = selector 選中的候選，`samples` = 其餘候選**依候選順序**。
- Judge Human-Spotting（2AFC）讀的是**第一個非空的 sample**（`score_method.py`：`next(x for x in samples if x.strip())`），
  不是 `greedy`；Lexical Separability、長度、Self-Repetition、語意距離等讀 `greedy`；Search-Move Mismatch 讀 greedy ＋ samples。
- **混淆**：reranker 決定哪一個候選被選走，也就**決定了 2AFC 讀到哪一個候選**（未被選中的第一個，通常是 slot 0；
  slot 0 被選走時就是 slot 1）。所以 v18 的 2AFC 變化同時來自 Planner 與 reranker 的選擇，而且 2AFC 衡量的是
  「我們**沒有**輸出的候選」。報告時必須寫明這一點；不可把 2AFC 的改善說成「輸出更像人」。
- **v18 不改 samples 的順序**（例如把 reranker 分數高的未選中候選排到前面）：那等於專門最佳化 2AFC 讀取的通道，是
  Goodhart；若要改，需要使用者另行決定（§17 Q3）。

## 7. Validation、選版與停止（§17 Q6：使用者決定）

### 7.1 資料與標籤

- Task 1：`validation_all`（4 段、20 則）；act 指標用 §1.1 的 validation 標籤（只用於本節）。
- Task 2：`validation`（4 段有需求標註）× seeds 0–7；episode 以 `clean_v18` 判定（§3.2）。
- 全部 validation 都經過含 reranker 的 selector。

### 7.2 排程

- **u0**（v18 run 自己的參考點，含 reranker；validation，不是 test arm）：Task 1 ＋ Task 2（seeds 0–7）。
- **每次 update 後**：Task 1（greedy ＋ P_end 探針，約 5 分鐘），記錄（每一項都是「越高越好」的分數）：
  - `stop_score` = mean(1 − (P_end − y)²)（t ≥ 2）；另記 `nll`（同 v17 B5）；
  - `act_score` = §3.1.2 r_act 的平均（greedy 計畫）；`act_entropy`；
  - `len_score` = §3.1.3 r_len 的平均；計畫長度的四分位距、被選訊息與真人字數的 W1；
  - `n_invalid`、`rerank_changed` 比例。
- **被選中的版本**：Task 2（seeds 0–7）。

### 7.3 選版規則（預先登記，只看 validation）**【待核准】**

1. **候選集合**：u ∈ {1..最後完成的 update}，同時滿足：
   - `nll(u) ≤ nll(u0) + 0.05`；`act_score(u) ≥ act_score(u0) − 0.02`；`len_score(u) ≥ len_score(u0) − 0.05`；
   - `act_entropy(u) ≥ 0.5 · act_entropy(u0)`（防 act 分布塌縮）；`n_invalid(u) ≤ n_invalid(u0) + 1`；
   - 該 update 沒有觸發長度漂移。（實作註記，稽核 B NIT 1：「觸發」指 §5 的長度漂移停止規則在 u 觸發，即 u−1 與 u 兩次
     update 的 mean(T − min(H, 10)) 都 < −1.0；單一 update 的訓練 rollout 漂移量的是 π_{u−1}，不據此排除 u。`v18_rules.drift_stop_at`。）
2. **分數** `J(u) = Σ_k w_k · (s_k(u) − s_k(u0))`，s = (stop_score, act_score, len_score)，w 同 §3.4 的 Task 1 權重（1, 1, 0.5）；
   三者都是越高越好，所以 J 越大越好。選 J 最大者，同分取較晚的 update。
   **J 與守門一律用名目權重**（1, 1, 0.5），即使 r_len 在訓練中被凍結（§4.1）：validation 仍要量長度，凍結只表示訓練沒有用它。
3. **Task 2 檢查**：選中者跑 Task 2 validation；需 `mean(T − min(H, 10)) ≥ −1.0`。
   不過 → 換 J 次高的候選再檢查一次；仍不過或候選集合為空 → final = u0（記為「GRPO 未採用」，reranker 仍照 §6.3 決定）。
   （預設不用 coverage 守門；若 §17 Q8 選擇保留 r_cov，另加 `coverage ≥ coverage(u0) − 0.05`。）
4. **提早停止**：連續兩次 update 的 J 下降（J(u) < J(u−1) < J(u−2)，u0 視為 J = 0），或長度漂移停止 → 停止訓練，照上面規則選版。
   訓練 reward 上升而 J 連續下降，是 reward 被鑽漏洞的訊號（§9），記錄為 `stop_reason = "val_decline"`。
5. **resume**：J、候選集合與提早停止的狀態**一律由 `validation.jsonl` 的列重算**（不另存狀態）；缺 validation 的 update 先補跑
   validation 再判斷。
- **限制要寫明**：validation 只有 4 段、20 則，這個規則主要是安全網，不是精細的模型選擇。
- `final.json`：`{final_update, stop_reason, selection: {candidates, J, guards, task2_check}, policy_sha, reranker_sha, adv_scales_sha, validated}`。

## 8. Test 評估（只評估 final；與既有的 u0、u5 比較）

- **我方 test**（`eval_test_rl.py`／`eval_test_boot.py`）：final（含 reranker）跑 Task 1（test_all 9 段：nll、bal_p、auc、term_f1、
  premature）與 Task 2（test × seeds 0–7：turn W1、|diff|、coverage〔診斷，judge 失敗為 null〕；episode 以 `clean_v18` 判定）。
  - 對照：v17 `runs/pend_f2_v17/test.jsonl` 的 SFT (u0) 與 final (GRPO) u5 列（已存在，不重跑）；paired bootstrap 以對話為
    單位（Task 2 依 (cid, seed) 配對），10000 次。
  - 報告用語：**v17 SFT (u0)**、**v17 GRPO (u5)**、**v18 (GRPO+R)**。
- **Benchmark 全套**（fold 2 test 側，scorer v3、本機 gpt-oss-120b effort low、judge cache；同 v17 的 `bench_eval_f2_v17`）：
  - `bench_tf_generate.py --reranker <json>` 產生 final 的 generations（method slot
    `sep_sim_pend_qwen3_4b_planner_ditto_8b_speaker_v18_fold_2_grpo_rerank_u<k>`）；
  - `bench_score_f2.sh` 計分（`--skip requirement_item_overlap,human_utterance_copying`、generations 與 corpus 同時切、
    `--dump-per-turn` 到 repo 樹外）；`score_stop.py` 與 termination probe（End-When-Done、Premature-End、Dialogue Length）。
  - 對照：`bench_eval_f2_v17/results` 的 u0、u5，以及 benchmark 既有方法在同一個 fold 2 test 側、同 scorer 版本與 judge 的結果檔
    （judge 或 scorer 版本不同的不放進同一張表，CLAUDE.md §1.4、§1.6）。
- **配對 bootstrap**：由 per-turn dump 以對話為單位重抽（9 段，10000 次），報告 v18 − u0、v18 − u5 的差與 95% CI、P(better)；
  Stop AUC 用 `score_stop.py` 內建的 paired conversation-cluster bootstrap。
- **預先登記的主要指標**（其餘為次要，只看方向）：Judge Human-Spotting（|acc − 0.5|，受 §6.5 的混淆影響，報告時並列說明）、
  Stop-Point AUC、End-When-Done、Dialogue Length（|· − 真人|）、Length W1。
  - Lexical Separability、MMD／Fréchet、Conversational-Act Rate Gap 在 fold 尺度不可信（CLAUDE.md §1.3 第 4 點），只列不判。
- **統計力**：test 只有 9 段、40 則。大多數差異的 CI 會跨 0；結論只能寫「方向一致／不一致」，不可宣稱顯著。
- **不做**：reranker-only、GRPO-only、在 test 候選上離線重選（會是 test 偷看）。只報告 `rerank_changed` 的比例（描述性）。
- **歸因限制**：v18 與 u0／u5 的差異是「reward、advantage 與 reranker」的合併效果，無法分開（使用者已接受）。

## 9. Goodhart 風險與對策

### 9.1 量尺分離

| 用途 | 我方訓練／選擇用的量尺 | benchmark 的評估量尺 |
|---|---|---|
| act | 我方標註器：gpt-oss-120b ＋ **我方 L2 分類（5 個非結束 move）**，只標 train_all | benchmark 標註器：自己的 codebook、6 類 L1（QUERY…CLOSE）＋ Stolcke 15 類 |
| 擬人度 | SimCSE-PCA ＋ 5 個風格特徵的成對 LR，只用 train_all | 2AFC LLM judge（無上下文）、TF-IDF LR（5-fold AUC） |
| 長度 | 計畫長度 vs train 真人字數（帶內滿分） | 被選訊息的 W1、KS、IQR、histogram JSD |
| 結束 | P_end 的 Brier（train 標籤） | score_stop（AUC、AP）、termination probe |
| 輪數 | 每個 train 情境的 \|T − min(H, 10)\| | free-running 平均輪數 |
| coverage | 預設**不用**（只做診斷） | Task-2 coverage（benchmark Ledger） |

- **共同點要承認**：我方標註器與 benchmark judge 是同一個 gpt-oss-120b（prompt 與分類不同）；兩者的系統性偏誤可能同向，
  所以 act 指標的進步不一定是「更像人」。報告時明寫。
- 5 個風格特徵和 benchmark 的 report-only「length/punctuation/uppercase 7-feature LR」有部分重疊；該 report-only 指標不作為 v18 的證據。

### 9.2 對策

1. **reward 只用 train 側 gold 與環境回饋**；validation 標籤只進 §7；test 不標註、不偷看；coverage 預設不當 reward。
2. **proper scoring rule**（r_stop、r_act 都是 Brier）：無法靠誤校準拿高分（相對於其標籤來源）；r_len 在帶內不施壓。
3. **KL 0.01 到 u0**、≤ 5 次 update × 8 步、lr 1e-5、advantage 大小校準到 v17：限制偏離幅度
   （reward 過度最佳化隨偏離距離惡化，Gao et al. arXiv 2210.10760）。
4. **reranker 不當 reward**：同一個判別器若同時是 Planner 的 reward 與 selector，Planner 會學著產生騙過它的計畫。
5. **選版只看 validation**，並監看「訓練 reward ↑、validation J ↓」的背離（§7.3 提早停止）。
6. **塌縮監看**：act 熵、計畫長度 IQR（§5、§7.3 的守門）。

## 10. 旗標、SPEC gate、provenance

- **新旗標與 SPEC 值**（不同就需要 `--ablation`）：
  - `--init-adapter runs/pend_f2_v17/ckpt/u00000`（必要，sha 鎖定）；不再有 `--sft-*`；
  - `--task1-positions all_t1`（t = 1..n；v17 的 argparse 只接受 `all`，§18）、`--task1-G 4`、`--task1-convs 8`；
  - `--reward-task1 multi`、`--w-stop 1.0`、`--w-act 1.0`、`--w-len 0.5`、`--w-fmt 1.0`、`--len-band 0.6667`；
  - `--reward-task2 v5`（r_turn ＋ r_fmt2）、`--w-turn 1.0`、`--w-fmt2 1.0`、`--w-cov 0`（§17 Q8）、`--stop-credit 0`；
  - `--adv-norm fixed_u0_tokenw`、`--adv-z-clip 3`、`--adv-target-task1 0.0197`、`--adv-target-task2 0.0762`（token 加權 τ，§4.3）、
    `--adv-min-spread-groups 3`（Task 1 各成分）、`--adv-min-spread-groups-task2 2`（r_turn，§4.1；兩者皆受 SPEC gate 與 verify 檢查）、`--adv-min-scale 0.01`（低於即凍結，§4.1）、`--fmt-scale 0.5`；
  - `--act-labels <train labels>`、`--act-labels-val <val labels>`（sha 記錄；train 檔的 cid 必須 ⊂ train_all）；
  - `--selector borda_rerank`、`--reranker <json>`、`--rerank-topk 2`；
  - `--lr 1e-5`、`--kl 0.01`、`--aux-weight 0.5`、`epochs 2`、`minibatches 4`、`--updates 5`、`--controller fixed`；
  - `--val-every 1`（Task 1）、`--val-seeds 0..7`（u0 與選中者的 Task 2）、`--length-drift-margin 1.0`、`--gpu-budget-gib 43`。
- `config_record["spec_version"] = "v18"`；新參數都不加入 `RESUME_MAY_CHANGE`（`gpu_budget_gib` 等資源設定照 v17）。
- `meta()`／`manifest()` 另記：init adapter sha、labels sha 與 meta（κ、缺票數）、reranker sha 與 LOCO 指標、`adv_scales.json` 的 sha、selector 設定。
- 程式快照 `pend_v18` ＋ sha 清單；中途修程式一律 `--resume --allow-code-change`，**不重新開始**（使用者規則）。

## 11. verify（`verify_pipeline.py` 新增 v18 分支；v16／v17 分支保留）

- **起點**：u0 sha = v17 u0；`ckpt/u00000` 是一般檔案（不是 link）；ref = u0；`ckpt` 不含 `ref/`；沒有 SFT 產物。
- **標籤**：reward 用的標籤 cid ⊂ train_all、∩ forbidden = ∅；validation 標籤只出現在 validation 紀錄；標註 prompt 與 benchmark
  codebook 無 8-gram 重疊；沒有任何 reward／標籤／reranker 程式讀過 `$BENCH/instruments/` 或 `$BENCH/data/`；
  預設設定下 coverage 沒有進入任何 reward 或選版。
- **Task 1 群組**：t = 1..n（扣掉截斷）；每組 4 樣本；t = 1 的狀態以 `valid_t1` 判定、值遮罩通過解碼核對、沒有 r_stop、
  沒有 aux 樣本；**由紀錄的原始 JSON 以 §3.1.2 的完全相同規則**（`acts.normalise`、丟棄規則、
  同 move 相加、均勻後備、Other／Complete 處理）重算 r_act，以 §3.1.3 的取筆規則重算 r_len，並重算 r_stop、r_fmt；
  不適用的分量沒有進入群組平均；剔除樣本沒有 advantage。
- **advantage**：以 `adv_scales.json` 的 `batch_sha256` 找到量測批次的列，由這批列重算 s_k、凍結清單、τ(κ=1) 與 κ 一致
  （update 1 中止過時，舊檔 `adv_scales.aborted_*.json` 的指紋對應被重產前的列，且未被用於任何梯度步）；每次 update 由紀錄重算 c、z（含截斷）、A（含 κ），並重算 τ_src 與 §4.3 的警示，
  與 learner 用的值一致（容差 1e-6）；Task 1 的值 token（含 t = 1）advantage = 0；r_stop 只在前綴；note token = 0；Task 2 的 stop credit = 0。
- **Task 2 乾淨 episode**：進入群組的 episode 都滿足 `clean_v18`（§3.2）；被剔除的都有記錄的原因；judge 失敗不構成剔除（預設設定）。
  - 新增 `pend.clean_v18_flag`：由 `episode_counters`、emitted_capped、compacted、輪數 > 0 重算 `clean_v18`，放在既有的
    `pend.clean_flag`（`verify_pipeline.py` 約第 562–567 行，保留）旁邊；
  - v18 run 的長度漂移重算（約第 1218–1236 行，目前用 `episode["clean"]`）與評估的 `eval.clean`（約第 1617–1621 行）改用 `clean_v18`；
  - validation／test 的 coverage 平均只取非 null，缺值數與紀錄一致。
- **reranker**：每一步的選擇可由 Borda 分數、top-2 規則與 `rerank_scores` 重算；reranker 檔 sha 一致；訓練資料 cid ⊂ train_all；
  LOCO 的 PCA 是逐折擬合（由 json 的逐折紀錄檢查）。
- **更新設定**：每次 update `optimizer_steps == 8`、KL 0.01、lr 1e-5、aux 0.5。
- **選版**：由 validation 紀錄重算候選集合、J、Task 2 檢查、提早停止與 final；停止原因一致。
- **test**：只有 final 一個政策；對照列讀自 v17 的檔案（sha 記錄）。
- **Ditto 架構**：同 v17 S12（speaker = Ditto-8B、沒有 end-token 旗標、arm = pend、tree = e1r_cf19400）。

## 12. 測試（dry-run，無 GPU）

- reward：r_act（丟棄規則、同 move 相加、均勻後備、Other／Complete、t = n 跳過、缺票）、r_len（取筆、帶、缺欄位、Other）、
  r_fmt、r_turn、r_fmt2 的單元測試。
- advantage：固定尺度的量測（只用有 spread 的群組、< 3 群組（r_turn < 2）或 < 0.01 即凍結、r_stop／r_act／r_turn 凍結時拒絕訓練、
  fmt 固定 0.5）、z 截斷與上界、token 加權 τ 與 κ 校準（τ(κ=1) = 0 時停止、κ₂ 低信心標記）、批次指紋、update 1 中止後重量測、
  u1 存在後拒絕重算、不適用分量、全零群組跳過、token 遮罩（前綴、t = 1 的值、note、Task 2 含值 token）、resume 讀回同一組尺度。
- turn 1：`task1_sample` 在 t = 1 不 raise、`valid_t1`、值遮罩的解碼核對、`task1_samples` 依 t 分流、`aux_examples` 跳過 t = 1、
  `score_task1` 不算 P_end、`run_conv` 處理 n = 1。
- Task 2：`clean_v18`（judge 失敗不剔除、R0 截斷／空白與 emitted_capped 剔除），validation／test 也用它（coverage 為 null 不重跑）；
  verify 的 `pend.clean_v18_flag`、v18 的漂移重算與 `eval.clean`；`pend.clean_flag` 補上 `judge_error` 後的回歸測試。
- turn 1：end_session 缺少或無效 → 無效（r_fmt = 0）。
- selector：top-2（含同分）、reranker 打破平手、guard 全滅後備、紀錄欄位。
- 標註器：8-gram 檢查、cid 檢查、空白／無法解析的重試與計數、5% 停止、快取重用；
  reranker：去重、LOCO 切分不跨對話、逐折 PCA、門檻判定、bootstrap CI。
- 選版規則、提早停止、由 validation.jsonl 重算（resume）、final.json；verify 抓得到竄改（標籤 cid、r_act 重算、advantage、選擇、final）。
- 既有測試全部通過；v17 專屬測試明確釘住 v17 設定。

## 13. GPU／時間預算（cfda5，2 × RTX PRO 6000；與他人共用，用 gpu_holder3 ＋ start_servers6）

依據 v17 實測：每次 update 27–55 分鐘（learner 7–17、Task 1 群組 9–16、Task 2 rollout 10–21 分鐘）；test 評估約 2 小時。
t = 1 使 Task 1 決策點多約 27%。以下各步**依序**執行（每一步依賴前一步）：

| 階段 | GPU | 估計時間 |
|---|---|---|
| 實作（sub-agent）＋ dry-run 測試 | — | 4–5 h |
| ≥ 3 個獨立稽核 ＋ 修正 | — | 1.5–2 h |
| 標註 train_all 79 則 ＋ validation_all 20 則 × 3 票 | gpt-oss | 15 min |
| reranker 負例：u0 在 train_all 的 TF 執行 × 2 seeds | Planner vLLM ＋ Ditto | 30–40 min |
| u0 在 validation_all 的 TF 執行（§6.3 門檻用的候選） | Planner vLLM ＋ Ditto | 10 min |
| reranker 擬合 ＋ LOCO ＋ 門檻 | CPU | 10 min |
| GPU smoke（1 次 update，縮小規模） | 2 GPU | 30 min |
| u0 validation（Task 1 ＋ Task 2 × 8 seeds） | 2 GPU | 30 min |
| GRPO ≤ 5 updates（每次約 30–60 分鐘） | 2 GPU | 2.5–5 h |
| 每次 update 的 Task 1 validation ＋ 選中者的 Task 2 validation | 2 GPU | 50 min |
| test（final：Task 1 ＋ Task 2 × 8 seeds） | 2 GPU | 1–1.5 h |
| benchmark 生成 ＋ 計分（final 一個 slot） | 1 GPU ＋ gpt-oss | 1.5–2 h |
| **合計（無 GPU 競爭時）** | | **約 16–20 h** |

- 標註與 reranker 資料可在稽核期間先跑（只依賴 u0 與既有程式）。
- 超時的退路（依序）：Task 1 validation 改為每 2 次 update 一次；`--updates` 降為 4（需使用者同意，屬 `--ablation`）。

## 14. 風險

1. **資料極少**：正例 79 則、17 段；reranker 可能只學到 u0 候選的表面特徵；act 標籤有雜訊（以軟標籤與 κ 監看，κ 只代表自我一致性）。
2. **Stop 再退步**：v17 的 Stop AUC 0.80 → 0.65。v18 讓 r_stop 只走前綴、advantage 大小對齊 v17，並以 validation nll 守門；
   但推理欄位同時被 act、長度的 advantage 拉動，結束訊號仍可能被稀釋，而 validation 只有 4 段。
3. **分布塌縮**：r_act、r_len 若過度最佳化，act 熵與長度 IQR 會下降（Kirk et al. arXiv 2310.06452）；以守門與 KL 控制。
4. **2AFC 的混淆**：reranker 決定哪一個候選留在 samples 給 2AFC 讀，2AFC 衡量的是我們沒有輸出的候選（§6.5）；
   2AFC 的變化不能解讀為輸出更像人。
5. **同一個 gpt-oss-120b** 同時是我方標註器與 benchmark judge。
6. **reranker 的分布漂移**：用 u0 的候選訓練，GRPO 後候選分布會變；不重訓（預算）。
7. **固定尺度的代表性**：s_k 與 κ 只在 u0 的第一批 rollouts 上量一次（約 30–40 個 Task 1 群組、4 個 Task 2 群組）；
   Task 2 的 κ₂ 估計尤其粗（恰好 2 個群組有 spread 時標記低信心）；只有 r_len 可被凍結，r_stop／r_act／r_turn 被凍結則不訓練。
8. **統計力**：test 9 段；validation 4 段；結論只能談方向。
9. **歸因**：與 u0／u5 的差異是合併效果（使用者已接受）。
10. **預算**：GPU 被他人佔用時，start_servers6 會退讓重試；以 §13 的退路處理，不搶他人的 GPU。
11. **Task 2 的 turn-1 值 token**：`rl_masks` 在 t = 1 不定位值，所以 Task 2 的 A_ep 會落在 turn 1 那個被忽略的 end_session 值 token 上
    （該值不影響任何行為，梯度只是雜訊）；與 v17 相同，照實記錄，不修。
12. **reranker 經 Implicit Profile 進入 Planner 的 prompt**：Task 1 的 teacher-forced 執行在 **train** 對話上使用 reranker，而 reranker
    就是用這些對話訓練的（樣本內）；被選中的訊息會透過 Implicit Profile 的 `measured_diff`／profile_note（「上一則我們產生的訊息
    哪裡寫錯」）回饋進 Planner 的 prompt，所以訓練時 Planner 看到的 prompt 分布與 validation／test（樣本外）略有不同。記錄，不修。
13. **Task 1 在損失分母中的比重變大**：t = 1 的群組（第一則真人訊息通常最長、計畫也較長）讓 Task 1 的 token 在 minibatch 分母中
    的比例高於 v17，Task 2 每個 token 的有效權重因此略降；§4.3 的 RL／aux 梯度比與 τ 警示會顯示這個偏移。

## 15. 依據（只列有把握的 arXiv id；不確定的已標註）

| 設計 | 依據 |
|---|---|
| GRPO、群組置中 | DeepSeekMath（arXiv 2402.03300）、Dr. GRPO（arXiv 2503.20783）、DAPO（arXiv 2503.14476） |
| 分量分開正規化 | **受 GDPO 啟發**（NVIDIA 2026，arXiv 2601.05242 **〔id 未確認〕**；GDPO 在群組內逐 reward 正規化、再對加總做批次正規化）；v18 改用 u0 上量一次的固定尺度（§4.1） |
| 多個 reward 以固定權重相加 | Fine-Grained RLHF（arXiv 2306.01693） |
| Brier 作為 reward | RLCR（arXiv 2507.16806）；proper scoring rule（Gneiting & Raftery, JASA 2007） |
| act 分布用 verbalized sampling | Verbalized Sampling（arXiv 2510.01171） |
| 判別器重排 | Residual EBM（Deng et al., arXiv 2004.11714） |
| best-of-n 的 KL 參考值 | Beirami et al.（arXiv 2401.01879）（此處只作啟發式） |
| 過度最佳化與早停 | Gao et al.（arXiv 2210.10760） |
| RL 降低多樣性 | Kirk et al.（arXiv 2310.06452） |
| lr、KL、少量步數 | 沿用 v17 §11（LoRA Without Regret、USP、AT-GRPO **〔id 未確認：v17 寫 2602.08533，稽核者記得 2510.11062〕**、Open-RS arXiv 2503.16219、Tina arXiv 2504.15777） |
| 使用者模擬器 | UserLM（arXiv 2510.06552）；USP（implicit profile，arXiv 2502.18968 **〔id 未確認〕**）；UserRL、MUSE **〔未查證，不引用數字〕** |

- **計算量對照**：上列論文的 RL 都是數百到數千步；v18 是 ≤ 5 × 8 = 40 步的 LoRA 試驗，預期只能看到方向，不能期待論文量級的效果。

## 16. 流程

1. 使用者核准本文件（含 §17 的決定）。
2. 實作（sub-agent，依 §18 清單）＋ dry-run 測試全部通過。
3. **多個**獨立 sub-agent 稽核（spec 對照程式、Ditto／pend 執行路徑與 selector、reward／advantage 重算、verify 與測試、
   資料邊界：train／validation／test 與 benchmark instruments／data），重複修正直到沒有發現。**訓練前必須全部 CLEAN。**
4. 標註（train_all、validation_all）→ reranker 資料（train_all）與門檻候選（validation_all）→ 擬合 → §6.3 門檻。
5. 程式快照 `pend_v18`；GPU smoke（P_end、各分量、固定尺度與 κ、token 加權 τ 落在 v17 範圍、advantage 重算、t = 1 的值遮罩、
   reranker 選擇、一次 update 8 步）；
   通過後跑 fold 2，每 10 分鐘監看。
6. 選版 → test → benchmark → 配對 bootstrap → 報告。

## 17. 要使用者決定的問題

- **Q1 act gold**：用我方標註器（gpt-oss-120b ＋ 我方 L2 分類、3 票軟標籤）標 train_all，可以嗎？要不要先看 20 則的審閱表？
- **Q2 權重與長度帶**：w_stop 1、w_act 1、w_len 0.5、w_fmt 1、w_turn 1、w_fmt2 1；長度 1.5 倍內滿分（ρ₀ = 2/3）。
- **Q3 2AFC 通道**：維持「samples = 未選中的候選、依候選順序」（不依 reranker 重排），並在報告中說明 §6.5 的混淆？
- **Q4 reranker 的位置**：Borda 前 2 名再由 reranker 選（本草案），或 reranker 當 Borda 的第 3 個訊號？
- **Q5 Task 2 reward**：以每個情境的 |T − min(H, 10)| 取代 v17 的分布 reward v4（log p_h − log q），確認？
- **Q6 選版**：v17 是「固定取最後一版」；v18 改為 §7.3 的 validation 守門＋J 選版＋提早停止（可能選回 u0）。或維持 v17 的固定最後一版？
- **Q7 候選數**：維持 4 個候選（NSAMP 3）。增加到 8 會讓 reranker 有更多空間，但 Ditto 成本約加倍，且改變 benchmark 的取樣設定。
- **Q8 coverage（稽核 BLOCKER 1）**：Ledger 讀 benchmark 的 `data/req_shards_v1.json`，是 benchmark 的 Task-2 coverage 量測工具。
  - **(a) 不用（本草案預設）**：Task 2 只剩 r_turn ＋ r_fmt2；coverage 只作診斷。後果：完全符合「reward 不用評估儀器」；
    但少了「別在 assistant 還沒交付前就結束」的訊號，Planner 可能更早結束（v17 的對話已偏短：4.15 vs 真人 5.6）；
    r_turn 與 validation 的長度守門部分補償。
  - **(b) 保留 r_cov（w 0.5，如 v16／v17）**：保留防早退的訊號；但等於用 benchmark 的 coverage 工具當 reward，Goodhart；
    fold 計分目前不輸出用它的指標（需求條目吻合家族一律 `--skip`），所以不會直接灌水已報告的數字，但要在報告中宣告，
    且日後若 benchmark 加入 Task-2 coverage 指標，v18 的數字對它不是獨立量測。選 (b) 時 §7.3 加回 coverage 守門。
- **Q9 rev 3／rev 4 的新數值**：z 截斷 ±3（取代 rev 2 的 |A| 截斷）；分量有 spread 的群組 < 3（r_turn < 2）或尺度 < 0.01 即凍結
  （r_stop／r_act／r_turn 凍結則不訓練，只有 r_len 可凍結）；κ 以 token 加權 τ 對齊 v17（Task 1 0.0197、Task 2 0.0762），無後備；警示門檻（τ 超過 v17 範圍 2 倍、
  RL／aux 梯度比 > 8、rl_grad_norm > 0.025）。

## 18. 實作清單（檔案層級；缺一項即未完成，稽核逐項核對）

1. `sep-sim/label_acts.py`（新）：我方標註器、軟標籤、κ、快取、cid 檢查、benchmark codebook 8-gram 檢查；completion 上限 2048、
   空白／無法解析重試 3 次、缺票計數、有效票 < 2 的處理、失敗率 > 5% 停止。
2. `sep-sim/rl_reward.py`：`task1_components()`（r_stop、r_act、r_len、r_fmt、適用性與狀態；r_act 依 §3.1.2 逐條實作）、
   `reward_v5()`（r_turn、r_fmt2；`w_cov` 預設 0），純 python。
3. `sep-sim/rl_algos.py`：`fixed_scale_advantages()`（群組置中、固定 s_k、z 截斷、κ；凍結分量 z = 0）、`measure_scales()`
   （§4.1：只用有 spread 的群組、凍結規則、批次指紋）、`token_weighted_tau()`（§4.3，與 learner 用同一個 `token_advantages`）、
   `calibrate_kappa()`（無後備：τ(κ=1) = 0 即停止；含 κ₂ 低信心標記）、`plan_mask_of()`、`token_advantages` 的 `A_pre／A_plan／A_ep` 分支。
4. `sep-sim/task2_env.py`：`task1_sample`（約第 1317 行）在 t = 1 不 raise；新增 `valid_t1`（不受第 1340–1341 行
   `end_session_t1_ignored` 的限制）；`rl_masks`（第 178–180 行）在 t = 1 改為直接呼叫 `planner.stop_mask` 取得 `value_mask_t1`，
   保留第 181–187 行的解碼核對；回傳原始 `act_distribution` 與各 move 長度；`episode_clean` 之外新增 `episode_clean_v18`
   （§3.2：只看 R0 截斷／空白、emitted_capped、compacted；judge 計數只記錄）；
   selector `borda_rerank`；每步記錄 `borda_index`、`rerank_scores`、`rerank_changed`；`describe()` 加 reranker sha 與設定。
5. `sep-sim/style_select.py`：`select_rerank()`（Borda top-k ＋ reranker）、`HumanLikenessScorer`（讀 json、SimCSE-PCA ＋ 風格特徵）。
6. `sep-sim/train_reranker.py`（新）：由 u0 的 train_all TF 執行建成對資料（去重、有效樣本數）、LOCO（逐折 PCA）選 C、Platt、
   validation_all 門檻與 cluster bootstrap CI、輸出 json。
7. `sep-sim/train_planner_rl.py`：`SPEC_VERSION = "v18"`、`SPEC_V18`；`--init-adapter`（無 SFT；**複製**到 `ckpt/u00000` 並核對 sha）；
   `meta()` 第 698 行的 `assert init_adapter is None` 改為 v18 assert 存在且 sha 相符；`--task1-positions` 新增 `all_t1`
   （目前第 1723 行只接受 `all`）；`run_conv` 不再跳過 n < 2；`task1_samples` 第 1202 行的 `assert t >= 2` 改為依 t 分流；
   `aux_examples`（第 1249–1256 行）跳過 t = 1；`score_task1` 對 t = 1 不算 P_end；第 1298 行的群組過濾改用 `episode_clean_v18`
   （§17 Q8 選 (b) 時用 v17 的定義）；多分量 reward；`adv_scales.json` 的量測、批次指紋、凍結、update 1 中止時改名保留並重量測、
   `ckpt/u00001` 存在後拒絕重算、resume 讀回；τ 與 §4.3 警示的紀錄；
   Task 2 reward v5、stop credit 0；validation 每次 update 的新指標；§7.3 選版、提早停止與 resume 重算；`final.json` 擴充；provenance。
8. `sep-sim/eval_test_rl.py`／`eval_test_boot.py`：v18 只評估 final；對照列讀 v17 的 test.jsonl（sha 記錄）；新的報告用語。
9. `sep-sim/bench_tf_generate.py`：`--reranker`、v18 method slot、provenance 記 reranker sha；samples 的對應不變。
10. `sep-sim/verify_pipeline.py`：v18 分支（§11），r_act／r_len 以與 `rl_reward` 完全相同的正規化規則重算（共用函式，不另寫一份）；
    新增 `pend.clean_v18_flag`（保留 `pend.clean_flag`）；v18 run 的漂移重算（約第 1218–1236 行）與 `eval.clean`（約第 1617–1621 行）
    改用 `clean_v18`；**修既有 bug**：`pend.clean_flag` 的預期值（約第 564–566 行）漏了 `judge_error == 0`，而 `task2_env.episode_clean`
    （約第 202–204 行）有這一條，補上（v16／v17 舊 run 重跑 verify 時，若因此出現新的 FAIL，照實記錄、不改舊 run）。
    `eval_test_rl.py` 與 validation 的重跑規則改為只看 `clean_v18`。
11. `sep-sim/test_v18.py` 等測試（§12）；`smoke_v18.py`（GPU smoke）。
12. `ops/`：標註、reranker、訓練、test、benchmark 的執行腳本（gpu_holder3 ＋ start_servers6、OOM 重試、`--allow-code-change` 續跑、
    不搶他人 GPU）；程式快照 `pend_v18` 與 sha 清單。
