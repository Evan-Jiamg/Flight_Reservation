---

## 3. 方法（Methodology）

> 註：本節只寫**已實作且在正式 run 路徑上**的設計（v16，`sep-sim/` 程式與 `ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）；每個數值都對應一個程式常數或 SPEC 值（見本節末的註）。放進 2 頁模板時保留 3.2 前兩句、式 (1)(2)(3) 與 3.4 的選擇分數，其餘移到附錄或全文版。

### 3.1 任務與資料

我們依照賽道在對話式資料集搜尋語料上的兩個任務。**Task 2** 中，模擬器拿到 persona 與目標（主題、情境、使用者已知道的資料集），和一個 task agent 對話，直到它結束對話或達到 T_max = 10 則使用者訊息的上限。**Task 1** 中，模擬器以真實對話到第 t−1 則訊息為條件，產生第 t 則；同一次執行也得到它在每個真實回合「是否結束」的決定，我們把它當作結束決策來評分（這是我們自己的對應方式，不是官方的 Task 1 指標）。訓練時，task agent 與評分涵蓋率的需求帳本（requirement ledger）都是本機的 gpt-oss-120b。我們採用 benchmark 的 goal 與 persona 都不重疊的三折切分：每個 fold 中，訓練用 `train`（有需求標註的對話，用於 Task 2 rollout）與 `train_all`（全部訓練對話，用於 Task 1 結束群組、few-shot 範例池與真人長度分佈）；`validation` 只用來挑 checkpoint；`test` 在訓練結束後讀一次，做最終評估。validation 與 test 的對話都不會進入訓練、few-shot 範例池、長度分佈或控制器。

### 3.2 模擬器

**Planner。** 每一輪，Planner（Qwen3-4B-Instruct-2507 加上一個 LoRA adapter，是唯一被訓練的元件）讀取目標、目前為止的對話與它自己先前的筆記，寫出一份 JSON 計畫：一句對自己先前狀態的批評、一份文字化的 dialogue move 分佈（每個 move 附目標長度）、目標是否已達成（`goal_met`：yes / partly / no）、使用者還想要什麼，以及 `end_session`——若正在規劃的訊息是使用者的最後一則則為 true。第一則訊息不能結束對話。若 `end_session` 為 true，這則訊息的 act 採用 Planner 自己機率最高的 *Complete* 項目，並告訴 Speaker 這是使用者的最後一則訊息；訊息送出後 episode 結束。

**隱含 profile。** 從第 2 輪起，Planner 也會寫一段 `profile_note`：用一兩句話描述*這個人*的寫作方式（長度、語氣、大小寫、措辭）中，它前一次的輸出寫錯了什麼。Task 1 中，它比較自己對第 t−1 則的預測與真實的第 t−1 則；Task 2 中，它批評自己前一則訊息。筆記在整段對話中累積，每一輪都傳給 Planner 與 Speaker。

**Speaker 與 selector。** 凍結的 Speaker（Ditto-8B）每輪根據 Planner 的計畫、筆記，以及 k = 3 則寫作風格相同的使用者的真實訊息（取自 `train_all` 中其他 goal 與 persona 的對話），寫出四則候選訊息（一則 greedy、三則以 temperature 0.7、top-p 0.9 取樣）。違反守衛（例如連續照抄範例 8 個字以上）的候選會被剔除，再由「長度＋風格」的 Borda selector（與使用者自己先前訊息、或與範例的 SimCSE 相似度）挑出一則。Speaker 自己的 end token 被遮蔽，所以對話只會透過 Planner 的決定（或依 benchmark 慣例，一則空白訊息）結束。

### 3.3 以 GRPO 訓練 Planner

每次更新都以目前的政策跑兩種 rollout，兩者都從 vLLM 提供的 adapter 取樣。

**Task 2 群組。** 每次更新抽 4 個訓練情境，每個情境跑 G = 4 個 episode（Planner temperature 1.0）。一個 episode 的 reward 是

  R = w_cov · coverage + w_dist · [log p_h(T) − log q(T)] − Σ_k λ_k · rate_k ，  (1)

其中 T 是使用者訊息數（上限 T_max）；p_h 是該 fold `train_all` 中真人訊息數的 add-α 平滑分佈（α = 1）；q 是本次更新乾淨 rollout 的同樣平滑分佈；coverage 是 episode 結束時 task agent 已處理的使用者需求比例（由帳本 LLM 判斷）；rate_k 是每步「計畫無法解析」與「計畫撞到生成上限」的比例。因為 E_q[log p_h − log q] = −KL(q ‖ p_h)，長度項在對話長度的*分佈*與真人一致時達到最大，而不是讓每段對話都同一個長度。task agent 回覆被截斷、帳本判決遺失，或送出的訊息撞到上限的 episode，不會進入群組。

**Task 1 結束群組。** 每次更新抽 8 段 `train_all` 對話；每段中，真實的最後一則訊息，以及（對話有 n ≥ 3 則訊息時）一則較早的訊息（t ≥ 2）是決策點。在每個決策點，Planner 看到真實歷史並取樣 G₁ = 8 份計畫；計畫的 `end_session` 與真人在該點的實際行為一致時 reward 為 1，否則為 0（無效計畫也是 0）。若有 reward 變異的群組數少於抽到的群組數，就再從 `train_all` 抽還沒用過的對話（最多再 8 段），直到有效群組數恢復。

**Advantage 與 credit。** 群組內使用群組平均為基準、不除以標準差（Dr. GRPO）：A_i = R_i − mean_j R_j；reward 沒有變異的群組直接跳過。Task 2 的 advantage 拆成長度項造成的部分 S_i = w_dist · [log p_h(T_i) − log q(T_i)] 與其餘部分：

  A_i^stop = S_i − mean_j S_j ，  A_i^seq = (R_i − S_i) − mean_j (R_j − S_j) 。  (2)

A_i^seq 套用在該 episode 每一份計畫的每個生成 token 上（`profile_note` 的 token 除外），A_i^stop 只加在真實決策點（t ≥ 2、可解析、未撞上限）的 `end_session` 值 token 上。Task 1 群組的 advantage 只作用在 `end_session` 值的 token 上。

**目標函數。** 令每個 token 的重要性比 ρ = π_θ / π_old，一個 minibatch 的損失為

  L = (1/N) Σ_tokens [ −w · min(ρ·A, clip(ρ, 1±0.2)·A) + β · (e^{d} − d − 1) ] + L_aux ，  d = log π_ref − log π_θ ，  (3)

其中 N 是生成的 token 數；β = 0.04 是對起始政策 π_ref 的 k3 KL 估計量權重；w = min(π_old / π_vLLM, 2) 是截斷的重要性權重，用來修正以 vLLM server 取樣造成的差異；若 |log π_old − log π_vLLM| 的平均超過 0.1，該次更新會中止。輔助項

  L_aux = −(w_eff / N_aux) Σ log p_θ(y* | prompt, 自己的前綴)  (4)

在基本群組的 Task 1 決策點上，監督真人的 `end_session` 值 y*（N_aux 是這些計畫的生成 token 數）。其權重 w_eff = max(0.5, w_aux · a_u)：a_u 在驗證顯示結束決策已改善之前為 1（balanced end probability（§3.4）連續兩次驗證都比未訓練政策高出 0.10），之後在 10 次更新內線性遞減、趨向下限，所以監督權重永遠不低於 0.5。

**權重控制器。** 每 5 次更新，一個 LLM 控制器（gpt-oss-120b）讀取*訓練* rollout 的摘要統計，可以把 w_cov、w_dist、λ_unparsed、λ_hit_max_new 與 w_aux 各乘上 {0.5, 0.8, 1, 1.25, 2} 中的一個倍數，但不得超出固定的上下限（w_dist 只能從 1 往上調）；若固定權重的影子 reward 連續兩個視窗變差，就撤回改動。

**設定。** LoRA rank 16（α = 32）加在 attention projection 上；learning rate 2e-5；每次更新做一次 optimizer step，梯度範數裁切於 1.0；HF 模型是 learner，並重新計算 vLLM server 生成的每個 token 的機率。

### 3.4 驗證與 checkpoint 選擇

每 5 次更新，在該 fold 的 validation 對話上驗證一次政策：Task 2 用取樣的 Planner（temperature 0.7、seeds 0 與 1），Task 1 用 greedy。Task 1 在每個真實決策點 t 計算 Planner 結束對話的 teacher-forced 機率：以 greedy 計畫的前綴為條件，對 `end_session` 值計算 P_end = p(true) / (p(true) + p(false))，再算平衡分數

  bal_p = ½ · mean_{t = n} P_end + ½ · mean_{2 ≤ t < n} (1 − P_end) ，  (5)

它獎勵在真實最後一則結束、在那之前繼續（無效的點視為 P_end = 0）。checkpoint 分數為

  score = coverage − W1(模擬輪數, validation 真人輪數) + bal_p ，  (6)

連續兩次驗證沒有進步就停止訓練（最多 30 次更新）。由於每個 fold 的 validation 只有 4 段對話，候選 checkpoint 會先以 8 個 seed 重新驗證再選定；選定的 checkpoint 與未訓練政策接著在 test 上各評估一次。

> 註：
> - 式 (1)–(6) 對應：`rl_reward.reward_v4`、`turn_distribution`；`rl_algos.split_group_advantages`（std_norm False）；`TorchLearner.update`（clipped surrogate × TIS、k3 KL、除以生成 token 數、aux_backward）；`train_planner_rl.aux_weight / aux_annealed`；`task1_stop.task1_prob_metrics`；`validate()` 的 selection。
> - clip：每次 update 只做 1 epoch × 1 minibatch，所以第一步的 ratio 恆為 1（程式有斷言），clip 在實際上不作用；式 (3) 仍照程式寫出。
> - 「連續兩次驗證沒有進步就停、最多 30 次 update」與「8 seeds 重新驗證」是實驗腳本（`ops/v11ops/run_v16_formal.sh`、`run_v16_reselect.sh`）的規則，不是 trainer 內建；「8 seeds」是使用者 2026-09-27 核准的。
> - coverage 是 reward 的一項（D3，已揭露）：因此訓練後的 coverage 不能當成獨立的評估指標，報告時要說明。
> - Dr. GRPO 需要引文（TODO）。
