# SPEC v17：SFT 暖身 → 短程 GRPO（使用者 2026-09-30 核准）

**這是唯一的設計（不做多個 arm）。** 實作與稽核一律以本文件為準；本文件沒寫到的行為沿用 v16
（`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）。凡本文件與 v16 衝突，以本文件為準。
依據：五個 sub-agent 的文獻與程式碼審查（見 §11）；使用者核准的關鍵值：LoRA lr 1e-5、KL 0.01（參考＝SFT 後政策）、固定 5 次 update。

---

## 0. 架構前提（不可改）—— **Ditto 新架構，不是 UserLM**

- arm `pend`（E1.6 tree `trees/e1r_cf19400`）：
  - Planner = Qwen3-4B-Instruct-2507 + LoRA，是**唯一**被訓練的元件；
  - Speaker = **Ditto-8B**（凍結）+ Borda selector（長度＋SimCSE 風格），並有 Implicit Profile、few-shot（k=3、同風格、其他 goal／persona）。
- **Ditto-8B 沒有結束對話的 token**：
  - 對話只會因兩種情況結束：Planner 的 `end_session: true`，或被選中的 Speaker 訊息是空白（benchmark 慣例）。
  - **任何 UserLM 專用的機制（end-token 遮罩、`<|endconversation|>`、SEPSIM_ENDMASK 類旗標）都不適用，也不可被重新引入。**
- Planner 生成走 vLLM（runtime LoRA，名稱 `p<u>-<sha12>`）；HF 模型是 learner。
- R0（task agent）與 ledger judge ＝ 本機 gpt-oss-120b，port 8029。
- T_MAX = 10；第 1 則不能結束；profile_note token 不帶 sequence advantage（D4）。
- Speaker、selector、few-shot、implicit profile、prompt 內容 **全部不改**。v17 只改 Planner 的訓練流程與評估。

## 1. Stage 0：SFT 暖身（`Trainer.sft_stage()`，在 u0 之前）

### 1.1 資料

1. **對話與決策點**：該 fold `split["train_all"]` 的每一段對話（assert ∉ forbidden），每一個真實決策點 t = 2..n（n = 真人訊息數）。
   - 與 v16 相同：t 之前的歷史中有被截斷（emitted_capped）的訊息，就跳過該點並計數。
2. **prompt**：用**起始政策**（base＋零效果 LoRA）做 greedy teacher-forced 執行，取得每個 t 的 Planner prompt（`env.task1_prompts(cid)`）。
3. **每個決策點取 3 份計畫**：
   - 1 份 greedy（`task1_sample(G=1, T=0)`）；
   - 2 份 T=1、top_p=1 的取樣，seed 由 `seed_of(seed, "sft", cid, t, k)` 決定。
4. **產生樣本**：每份計畫若「決策有效且找得到 stop mask」，就用 `planner.stop_target(gen_ids, stop_mask, want_end = (t == n))` 產生一筆樣本：
   `{cid, t, n, real_final, kind: greedy|sample, prompt_ids, prefix_ids, target_ids, gen_len}`。
   - 無效或找不到 mask 的計畫不產生樣本，依原因分別計數。
5. **快取**：
   - 寫入 `sft_examples.jsonl`，並另存 `sft_examples_meta.json`（起始政策 sha、splits sha、code sha、各類計數）。
   - resume 時若 meta 相符就直接重用，不重新生成。
6. **base P_end 快取**：同一階段，對每個 train_all 決策點記錄起始政策的 teacher-forced P_end（greedy 前綴，同 `task1_end_probe`），寫入 `base_pend_train.jsonl`，供 §7 的門檻對照使用。

### 1.2 損失與最佳化

- **損失**：每個 minibatch 為 −Σ log p(target_ids | prompt ＋ prefix) ／ Σ gen_len（與輔助監督同一個式子，抽成共用函式）。
- **不做類別加權**：保持校準。依據：交叉熵是 proper scoring rule；加權成 1:1 會把隱含結束率推高、對話變短。
- **最佳化器**：獨立的 AdamW（不沿用、也不交給 GRPO 的 optimizer），`--sft-lr` 5e-5，minibatch 8 筆，每個 epoch 以 `seed_of(seed, "sft_shuffle", epoch)` 洗牌，`--sft-epochs-max` 3。

### 1.3 選 epoch（不是用訓練集判斷）

- **候選**：epoch 0（未 SFT）以及每個 epoch 結束時的 adapter。
  - 每個候選存到 `ckpt/sft_e<k>/adapter`，並由 vLLM 以 runtime adapter 服務。
  - epoch 0 使用零效果 LoRA 存成的 adapter。
- **評估**：在 `validation_all`（三個 fold 都是 4 段）的所有決策點上，以該候選的 greedy Task 1 執行計算 teacher-forced P_end（同 validate 的 Task 1 探針），記錄：
  - `nll` = mean −log p(真人標籤)，P 夾在 [1e-6, 1−1e-6]；
  - `bal_p`、`auc`、`n_points`、`n_invalid`。
- **選擇**：選 `nll` 最低者，平手取較早的 epoch。
- **記錄**：`sft.jsonl` 每個 epoch 一列，含 train loss、train P_end(final／nonfinal) 均值、上述 validation 指標，以及選擇結果。
- **產物**：
  - 選中的 adapter 存為 `ckpt/u00000`（policy_sha、LATEST 等與 v16 相同格式），即「SFT (u0)」。
  - 若選中 epoch 0，u0 就等於起始政策，照實記錄。
- **resume**：u0 已存在就絕不重跑 SFT；SFT 中途失敗則從快取樣本重跑 SFT（不寫 u0 與 LATEST，直到 SFT 完成）。

## 2. KL 參考政策 ＝ SFT 後的 u0

- **做法**：build 或 resume 時，把 `ckpt/u00000/adapter` 以名稱 `"ref"` 載入成**第二個凍結 adapter**（is_trainable=False）。
  - `_logp(reference=True)` 的流程：set_adapter("ref") → forward → set_adapter("default")。
  - 每次切換回 "default" 後 assert `trainable_names()` 不變，且 "ref" 的參數 requires_grad 全為 False。
- **checkpoint 只存 "default"**：`save_pretrained(selected_adapters=["default"])`；verify 檢查 adapter 目錄內沒有 `ref/`。
- **KL**：k3，`kl_coef` 0.01（取代 v16 的 0.04）。KL 以 u0 為參考，所以 SFT 學到的結束行為不會被拉回 base。

## 3. GRPO：每次 update 的內容

### 3.1 Task 2 群組（沿用 v16，只改列出的項目）

- 每次 update 抽 4 個 `train` 情境 × G=4 個 episode；Planner temperature 1.0（vLLM 強制 1.0／1.0）。
- reward v4：`w_cov·coverage + w_dist·[log p_h(T) − log q(T)] − λ·format`，初始權重皆 1。
- 其他與 v16 相同：stop credit（長度項 advantage 只加在 end_session 值 token）、note mask、Dr. GRPO（不除以標準差）、不乾淨 episode 不進群組。
- **控制器改為 fixed**（權重不動）。原因：5 次 update 內 LLM 控制器最多只會動一次。

### 3.2 Task 1 群組（新設計）

- **對話**：每次 update 以 `rng = Random(seed_of(seed, "task1", u))` 從 `train_all` 抽 `--task1-convs` 8 段。
- **決策點**：每段取**所有** t = 2..n（真實比例），截斷歷史的點照 v16 跳過並計數。
- **prompt**：目前政策的 greedy teacher-forced 執行（`task1_prompts`，與 v16 相同）。
- **取樣**：每個決策點 G₁ = `--task1-G` 4 份計畫，temperature 1、top_p 1，seed 同 v16。
- **reward**（Brier 式，proper）：
  - 樣本「決策有效且找得到 stop mask」→ learner（rollout 政策 π_old）在該樣本**自己的前綴**上做 teacher-forced 計算，得到 `P_end,i = p(true)/(p(true)+p(false))`（同 `end_prob`），`R_i = 1 − (P_end,i − y)²`，其中 y = 1[t == n]。
  - 樣本「決策有效但找不到 stop mask」（stop_mask_mismatch）→ **從群組中剔除**，計數，不產生梯度。原因：這是我們解析失敗，不該懲罰政策。
  - 樣本「無效」（未解析、撞到 max_new、end_session 值無效）→ `R_i = 0`（Brier 的最差值），計數。
  - `P_end` 與 `R` 寫入 `rollouts_task1.jsonl` 的每個樣本，resume 與 verify 都用這份紀錄。
- **advantage**：Dr. GRPO，`A_i = R_i − mean(R)`，只在剔除後的群組內計算；群組標準差 ≤ min_group_std 就跳過並計數。
- **advantage 作用的 token**：
  - 有效樣本：只加在 `prefix_mask`，也就是第一個 stop-mask token 之前的所有生成 token，再乘上 (1 − note_mask)；**不加在 end_session 值 token 上**（reward 已對兩個值取機率，值 token 由輔助監督訓練）。
  - 無效樣本：加在全部生成 token × (1 − note_mask)，作為格式懲罰。
- **拿掉 v16 的 refill（動態補抽）**：在連續 reward 下它幾乎不會觸發；相關程式、紀錄與 verify 都一併處理。

### 3.3 輔助監督（aux）

- **樣本**：每個決策點一筆（所有點，真實比例），取該點第一個「有效且有 mask」的樣本的 `stop_target(want_end = y)`。
- **權重固定為 `--aux-weight` 0.5**。**拿掉 D2 觸發與退火**，控制器也不調整它。
- **損失**：與 SFT 相同的式子，見 §1.2。

### 3.4 learner 更新

- **步數**：`epochs` 2 × `minibatches` 4 ＝ 每次 update 8 個 optimizer step（改 `ALGO_DEFAULTS`，不經 --config）。
- **clip**：0.2；**TIS**：cap 2，vLLM 與 learner 的平均 |Δlogp| > 0.1 就中止（同 v16）；`ratio_init` 只檢查第一步。
- **lr**：1e-5（使用者核准），AdamW（無 weight decay），梯度裁切 1.0。
- **aux 的分配**：aux 樣本平均分到每個 epoch 的 4 個 minibatch（每個樣本每個 epoch 用一次）；每一步的 aux 損失以**該 minibatch 的 aux gen_len 總和**正規化，梯度加到同一步的 RL 梯度上。
- **紀錄**：
  - `aux_p_correct_before` 在第一步之前量；
  - `rl_grad_norm`、`aux_grad_norm`、`kl`、`clip_frac`、`adv_abs_mean`（分 task1／task2 來源）記錄每一步的平均與最大值；
  - `optimizer_steps` 必須等於 8。
- 控制器（若未來改回 llm）的系統提示中「share one optimizer step」一句要改成符合實作的文字。

### 3.5 崩潰監看與停止條件

- **每次 update 都記錄**：
  - 各來源群組 reward 標準差的平均；
  - Task 1 樣本的結束比例（真實最後一則／較早位置）；
  - Task 1 的 P_end 均值（真實最後一則／較早位置）；
  - Task 2 的平均輪數與 `turn_hist`；
  - KL(q‖p_h)。
- **長度漂移停止**：該次 update 所有乾淨 Task 2 episode 的 `mean(T_episode − 該情境真人輪數)`，若連續 2 次 update < −1.0，就在該次 update 存完 checkpoint 後停止，停止原因記為 `length_drift`（寫入 run 紀錄，不是 crash）。
- **固定上限** `--updates 5`。**不用 validation 選 checkpoint**：最終版 ＝ 最後一個完成的 update（u5，或漂移停止時的那一個）。

## 4. Validation

- `load_split` 回傳 `validation_all`，並 assert：⊂ forbidden、與 train／train_all 不相交、⊇ validation。
- **u0（SFT 後）與最終版**：
  - Task 2 用 validation（有需求標註的）× **8 seeds（0–7）**；
  - Task 1 用 `validation_all`。
- **u1–u4**：只做 Task 1（`validation_all`，greedy＋探針，成本低），不做 Task 2。
- summary 記錄：Task 1 的 `nll`、`bal_p`、`auc`、stop F1（M2 對應）、premature；Task 2 的 turn_stats。
  - **不寫 selection_score**，拿掉 best.json 的選擇機制；保留 `task1_base` 作為比較基準，寫入 u0 的數值。
- **拿掉** `reselect()` 與 `--reselect-*`，並從 `RESUME_MAY_CHANGE` 移除。

## 5. Test 評估（`eval_test_rl.py` 與 `eval_test_boot.py`）

- **要評估的更新**：exactly `{0, final}`。final 讀自 run 的 `final.json`（新檔：最終 update、停止原因、policy sha）。
- 其他沿用：`--final` gate、provenance、test once、Task 2 用 test × 8 seeds、Task 1 用 test_all、paired bootstrap。
- **另外報告**（不重跑）：v16 已有的原始 base test 數字（`runs/pend_f2_v16/test_boot.txt` 的 u0 列），並在報告中標明為「base」。

## 6. 門檻對照（只做 Task 1，不訓練）

- **新工具 `threshold_control.py`**（純 CPU）：
  - 在 `base_pend_train.jsonl` 上擬合一個純量 logit 偏移 b，最小化 NLL：`P' = σ(logit(P_end) + b)`；
  - 套用到 base 在 test_all 的 end_probs（來自 test.jsonl／base 的 Task 1 探針），報告 nll、bal_p、auc，以及以 P' > 0.5 為決策的 stop F1 與 premature。
- **目的**：若只調門檻就追上 SFT 或 GRPO，就要照實報告。

## 7. 報告用語

- 一律寫 **base**（未訓練，即 v16 的 u0）、**SFT (u0)**、**final (GRPO)**、**threshold control**，不再用單獨的「u0」。

## 8. 旗標、SPEC gate、provenance

- **新旗標與 SPEC 值**（與 SPEC 值不同就需要 `--ablation`）：
  - `--sft-lr 5e-5`、`--sft-epochs-max 3`、`--sft-samples-per-point 2`（外加 1 份 greedy）；
  - `--task1-G 4`、`--task1-convs 8`、`--task1-reward brier`、`--task1-positions all`；
  - `--aux-weight 0.5`、`--kl 0.01`、`--lr 1e-5`；
  - `epochs 2`、`minibatches 4`；
  - `--updates 5`、`--val-every 1`（u1–u4 只做 Task 1）；
  - `--val-seeds 0 1 2 3 4 5 6 7`（只用於 u0 與 final 的 Task 2）；
  - `--length-drift-margin 1.0`、`--controller fixed`。
- **移除的旗標**：`--stop-sup-floor`、`--t1-trigger-margin`、`--stop-sup-anneal`、`--reselect-*`，以及舊的 `--task1-G 8` 規則。
- **provenance**：
  - `meta()` 與 `manifest()` 另外記錄 SFT 參數、`sft_examples` 的 sha、u0 的 sha、所選 epoch；
  - 新參數都不加入 `RESUME_MAY_CHANGE`。
- **程式快照**：code snapshot 新建 `pend_v17`，並附新的 sha 清單。舊 run（v16、v11）仍須能用 verify 通過，依 run_meta 的版本分流。

## 9. verify（`verify_pipeline.py`，依 run_meta 版本分流 v16／v17）

v17 必須檢查：
- **SFT 資料**：`sft_examples` 的對話都 ⊂ train_all 且不在 forbidden；決策點 = 所有 t=2..n（扣掉截斷）。
- **SFT 選擇**：由 `sft.jsonl` 重算選擇（nll 最低、平手取早），並確認 u0 的 sha = 所選 epoch 的 adapter sha。
- **ref adapter**：checkpoint 的 adapter 目錄不含 `ref/`；run_meta 記錄 ref = u0 的 sha。
- **Task 1 群組**：
  - 決策點 = 抽到的對話的所有 t=2..n；每組樣本數 = 4；
  - 每個有效樣本 `R = 1 − (P_end − y)²` 可由紀錄重算，P_end ∈ [0,1]，無效樣本 R = 0，剔除的樣本沒有進入 advantage。
- **advantage 位置**：Task 1 的 advantage 只落在 prefix_mask（有效）或全部 token（無效），都 × (1 − note_mask)；Task 1 樣本的值 token 上沒有 advantage。
- **aux**：每個決策點至多 1 筆、權重 0.5、每次 update 的 aux_n 與決策點數一致。
- **更新設定**：每次 update 的 `optimizer_steps == 8`；KL 係數 0.01；lr 1e-5；控制器為 fixed。
- **長度漂移**：由 updates 紀錄重算，和 run 紀錄的停止原因一致。
- **validation**：Task 1 的對話 == validation_all；u0 與 final 的 Task 2 seeds == 0..7；u1–u4 沒有 Task 2。
- **test**：更新集合 == {0, final.json 的 final}。
- **Ditto 架構**：run_meta 的 env 描述中 speaker = Ditto-8B、沒有 end-token 遮罩旗標、arm = pend。

## 10. 測試（dry-run，無 GPU）

- 擴充 FakeLearner／FakeEnv，支援：prefix_mask advantage、多步更新、SFT stub、P_end、ref adapter stub。
- **v17 每一項都要有測試**：
  - SFT 資料、選擇與 resume（u0 存在就不重跑）；
  - Brier reward 與剔除規則、advantage 位置、aux 分配、optimizer_steps；
  - 長度漂移停止、validation 排程、test 的更新集合、門檻對照工具、verify 抓得到竄改。
- 既有測試全部通過；v16 專屬的測試改成明確釘住 v16 設定，或針對舊 run 的 verify 分流。

## 11. 依據（摘要；詳見本 session 的 sub-agent 報告）

| 設計 | 依據 |
|---|---|
| SFT 學結束 | UserLM（arXiv 2510.06552）、AT-GRPO（arXiv 2602.08533） |
| 不加權、以 validation NLL 早停 | DPO／IPO／ORPO 的過擬合觀察、proper scoring |
| 不用 DPO | Razin et al. ICLR 2025（arXiv 2410.08847）、Smaug（arXiv 2402.13228）：單一 token 的偏好對會壓低正確 token |
| Brier reward | RLCR（arXiv 2507.16806） |
| on-policy 優於離線 | Tang et al.（arXiv 2405.08448） |
| Dr. GRPO、token 層級 | arXiv 2503.20783、DAPO（arXiv 2503.14476） |
| LoRA lr ≈ 10× 全參數 | LoRA Without Regret（Thinking Machines）、verl LoRA 指南 |
| KL 0.01 | USP、AT-GRPO |
| 固定少量步數 | Open-RS（arXiv 2503.16219）、Tina（arXiv 2504.15777） |
| 崩潰監看 | RAGEN（arXiv 2504.20073） |

## 12. 流程

1. 實作（sub-agent）＋ dry-run 測試全部通過。
2. **多個**獨立 sub-agent 稽核，重複修正直到沒有發現：
   - spec 對照程式碼；
   - Ditto／pend 執行路徑；
   - verify 與測試。
3. 程式快照 `pend_v17` 與 sha 清單；實驗腳本（佔位、servers、動態 GPU、OOM 重試、釋放；不搶他人正在用的 GPU）。
4. 等 cfda5 有空 GPU 時先跑 GPU smoke test，在 GPU 上實測 P_end、ref adapter 切換、SFT 一個 epoch、一次 update；通過後再跑 fold 2 試驗，每 10 分鐘監看。

---

## 13. 實作決定（實作前 gap check 的 10 個 BLOCKER、15 個 SHOULD；2026-09-30）

以下決定與 §0–§12 同等效力；與前文不一致時，以本節為準。

### BLOCKER
- **B1 起始政策可重現**：`setup_policy` 在 `get_peft_model` 之前 `torch.manual_seed(seed_of(seed, "lora_init"))`（LoRA B=0，A 由 seed 決定）；起始政策 sha 因此每次相同。
- **B2 固定 5 次**：`updates` 從 `RESUME_MAY_CHANGE` 移除；`updates != 5` 需要 `--ablation`；`final.json` 存在時 `run()` 拒絕再訓練（含 `--resume`）。
- **B3 vLLM 服務路徑**：`sync_generation_policy` 改成接受任意 adapter 路徑與標籤（`use_adapter(path, tag)`）。SFT 資料用 `ckpt/sft_e0`（起始政策存成 adapter）服務，每次生成都帶 content sha；verify 的 `adapter_name()` 同時支援 `ckpt/u%05d` 與 `ckpt/sft_e%d`。
- **B4 ref adapter**：`TorchLearner.save` 一律 `save_pretrained(..., selected_adapters=["default"])`。ref 在 optimizer 建立**之後**以 `load_adapter(u0, "ref", is_trainable=False)` 載入：
  - 新訓練時在 `sft_stage` 結束、u0 存好之後載入；resume 與 `eval_test_rl` 的 build 時也要載入；
  - assert optimizer 的參數集合與 trainable_names 不變；
  - `_logp(reference=True)` 以 `try/finally` 包住 set_adapter("ref") → forward → set_adapter("default")；
  - 不使用 per-forward 的 `adapter_names=`。
- **B5 validation NLL**：
  - `nll` 只算**有效**的點（未解析、截斷、找不到值 token 的點不計），另外報告 `n_invalid`；
  - SFT 選 epoch 先比 `nll`，再比 `n_invalid`（少者優先），再取較早的 epoch；
  - 欄位名稱統一為 `nll`（`task1_prob_metrics` 的 `logloss` 保留為別名，值相同）。
- **B6 門檻對照**：
  - 擬合 b 只用有效點（P_end 夾在 [1e-6, 1−1e-6]）；
  - base 的 test end_probs 在 fold 2 使用 v16 run 的 test.jsonl（u0 列）；
  - fold 0／1 由 `eval_test_rl --include-base` 以 `ckpt/sft_e0` 評估 base（Task 1 與 Task 2），fold 2 不需要（已有）。
- **B7 小群組**：剔除後少於 2 個樣本的 Task 1 群組跳過並計數（`n_groups_lt2`）；全部無效（R 全為 0、標準差為 0）的群組照 Dr. GRPO 跳過，不給格式懲罰，並計數（`n_groups_all_invalid`）。
- **B8 版本分流**：`config_record["spec_version"] = "v17"`；verify 以此欄位分流，沒有這個欄位且有 `task1_G` 的 run 走 v16 分支；`check_v16`、`check_rl_selection`、`check_selection`、0/1 reward 重算都只在 v16 分支執行。
- **B9 程式路徑**：trainer 只實作 v17（refill、D2、reselect、best.json、LLM 控制器的 v16 設定都從 trainer 移除）；verify 保留 v16 分支供舊 run 使用；v16 的 trainer 測試刪除或改成 verify 用的 fixture。
- **B10 test 工具**：`eval_test_rl.py` 與 `eval_test_boot.py` 改讀 `final.json`，更新集合 = {0, final}，不再讀 `reselect_*`。

### SHOULD
- **S1 每個樣本的 P_end**：
  - `task1_sample` 對每個「有效且有 mask」的樣本輸出 `target_true`、`target_false`、`prefix_ids`、`mask_ok`；env 不再計算 reward（改名為 `correct`，僅供記錄）；
  - tt 或 tf 為 None，或兩者的 prefix 不同 → 視同 mask 找不到，剔除並計數；
  - P_end 在 `Trainer.task1_rollouts` 以 learner（π_old）計算：在 `env.gpu_lock` 內、eval 模式，於 `append_jsonl` 之前寫入列中。
- **S2 advantage 組合**：
  - token 的 advantage = `adv·(1−note)` + `adv_prefix·prefix_mask·(1−note)` + `adv_stop·stop_mask`；
  - Task 1 不經過 `stop_credit` 分支；
  - note_mask 為 None 時視為全 0；
  - assert prefix_mask 與 stop_mask 不相交。
- **S3 步數**：樣本數 n < 4 時 minibatch 數 = min(4, n)，照實記錄；verify 要求 `optimizer_steps == epochs × min(minibatches, n)`；只有 aux 的 update 記為 1 步並標記。
- **S4 aux 分配**：
  - aux 另用 `seed_of(seed, "aux_mb", u, ep)` 洗牌，依序分成 4 份（數量 < 4 時有些 minibatch 沒有 aux）；
  - `aux_p_correct_before` 在第一步前以 no-grad 量測；
  - aux 與 SFT 共用模組函式 `value_nll_loss`。
- **S5 長度漂移**：真人輪數取 `min(human_turns, t_max)`；episode 集合 = 進入群組的乾淨 episode（`clean_eps`）；停止時寫 run_meta（kind `"stop"`）與 `final.json`。
- **S6 順序**：
  - 流程為 `one_update(u)` → 漂移判定 →（停止或 u == 5 時）寫 `final.json` → `validate(u, task2 = (u == 0 or u == final))`；
  - validation summary 的鍵包含 `task2`，所以 Task-1-only 的 summary 不能當成 final 的驗證；
  - resume 時由紀錄重算 final 狀態。
- **S7 SFT 的 resume**：
  - 非 resume 的啟動，若 `sft_examples.jsonl` 或 `ckpt/sft_e*` 已存在就拒絕；
  - `--resume` 且沒有 LATEST → 重用 SFT 快取繼續 SFT；
  - `save_checkpoint(0)` 移到 `sft_stage` 結束時，接著載入 ref。
- **S8 選 epoch 時兩端一致**：每個候選 k，learner 持有該候選權重，vLLM 服務 `ckpt/sft_e<k>`；Task 1 探針寫入 `sft.jsonl`；選定後 `learner.load_policy(ckpt/sft_e<k>)` 並核對 sha，再存 u0。
- **S9 固定控制器**：
  - `controller fixed` 為 SPEC 值（`llm` 需要 `--ablation`）；
  - `--intervention` 在 v17 不可用（直接拒絕並說明）；
  - `w_aux` 由 `--aux-weight` 設定；`aux_weight()` 回傳常數；
  - `check_history` 需要的鍵保留。
- **S10 預設值位置**：v17 的 lr 1e-5、kl 0.01 寫在 trainer 的 SPEC 表，不改 `rl_controllers.TRAIN_DEFAULTS`（避免牽動舊測試）。
- **S11 stop credit**：`--stop-credit 1` 保留為 SPEC 值，只作用於 Task 2。
- **S12 Ditto 的紀錄**：
  - `describe()` 新增 `speaker_path`、`speaker_class`（MRO）、`speaker_endconv_is_none`；
  - verify 檢查這些欄位，以及 arm_env 的 `SEPSIM_ENDMASK_RETRY/ENDGATE/KEEPEND/ENDSCORE == "0"`、`SEPSIM_END_PROBE == "0"`、tree = e1r_cf19400；
  - 文件註明：Ditto guardrail 會重抽空白（最多 6 次），selector 偏好非空白，所以結束實際上幾乎只來自 `planner_end` 或 `t_max`。
- **S13 final.json**：`{final_update, stop_reason: "max_updates" | "length_drift", policy_sha, validated: bool, time}`；`eval_test_rl` 要求 0 與 final 都有包含 Task 2 的 validation summary。
- **S14 smoke_v17.py**：檢查 P_end 與 `end_prob` 一致、ref 切換前後 trainables 不變、SFT 一個 epoch、一次 update 共 8 步、存檔沒有 `ref/`。
- **S15 統計欄位**：`t1_hist` 的 `acc` 改名為 `brier_mean`；新增 `p_end_final_mean`、`p_end_nonfinal_mean`、`n_dropped_mask`、`n_invalid`、`n_groups_lt2`；移除 refill 欄位；`adv_abs_mean` 分 task1／task2。

### NIT
- **N1**：SFT 的 3 份計畫以 `task1_sample(G=1)` 呼叫三次，seed 分別是 `seed_of(seed, "sft", cid, t, k)`（k=0 為 greedy）。
- **N2**：重複的樣本保留。
- **N3**：SFT 的梯度裁切 1.0、weight decay 0、forward mode 為 `train_nodropout`。
- **N4**：`turn_stats_seeds01` 的說明從「D5」改成「seeds 0/1 子集」。
- **N5**：所有 docstring 與提示文字中關於 D2、reselect、best 的描述更新。

### 實作補註（fix round 1，2026-09-30）
- **S15 鍵名（A-9）**：控制器 history 不可含有 `valid` 字樣的鍵（`rl_controllers.FORBIDDEN_KEY_PARTS`），所以 history 裡的
  `task1_train` 用 `n_not_decisions`／`n_groups_no_decision`；規格名稱 `n_invalid`／`n_groups_all_invalid` 放在 updates 列的
  `task1_stats`（verify 讀這一份）。
- **截斷歷史的跳過（C-S3）**：verify 只能檢查被跳過的決策點在 SFT 與 Task 1 都是後綴（t ≥ t0）；某則訊息是否 emitted_capped
  本身無法只靠紀錄重算（需要原始資料），這一點由 trainer 的 assert 與紀錄負責。

### 實作補註（fix round 2，2026-09-30）
- **fold 2 的 base（使用者決定）**：取代 §5 最後一點與 B6 中「fold 2 用 v16 run 的 test.jsonl（u0 列）」：所有 fold（包括 fold 2）都用
  `eval_test_rl --include-base` 以 `ckpt/sft_e0` 在 v17 定義下重新評估 base（fp32 value logits 的 P_end、human turns 以 t_max 截斷），
  寫入 `test_base.jsonl`；§6 的門檻對照一律用這份 v17 `test_base.jsonl`。v16 `runs/pend_f2_v16/test_boot.txt` 的 u0 列只當作
  標明的參考（「v16 base：bf16 P_end、未截斷 |diff|」），不再稱為 base。
- **只有一則訊息的對話**：沒有決策點（t = 2..n 為空）；`sft_examples_meta.json` 以 `n_by_conv` 記錄每段 train_all 對話的 n，
  verify 只接受 n < 2 的對話沒有 SFT 點或 Task 1 列。

### 實作補註（先佔資源，使用者 2026-09-30：「不能被搶卡，先佔資源」）
- 訓練／test／smoke 行程在 build 結束時（所有模型都已在訓練 GPU 上）以 `rl_algos.reserve_gpu_budget` 讓 PyTorch caching allocator
  持有 `--gpu-budget-gib`（SPEC 43，佔位程式交出 45 GiB，留 2 GiB 給 CUDA context）減 0.5 GiB：先配置再釋放（不 empty_cache），
  釋放的區塊仍由本行程保留、可供自己的 tensor 使用；expandable_segments 下同樣成立（只有 empty_cache 或 allocator 內部 OOM 重試
  才會把快取還給驅動程式）。每個階段（SFT、每次 update、每次 validation、test 的每個 policy）都重新檢查、補回；本程式的執行路徑
  沒有任何 empty_cache。
- 若預算已被別人拿走：印出「GPU budget not available … CUDA out of memory」並以 exit code 75 結束，run 腳本當成 OOM 重試（grep 也含
  "GPU budget not available"）。`--gpu-budget-gib` 是資源設定：可在 resume 時改（RESUME_MAY_CHANGE），正式 run 用 43 以外的值需要
  `--ablation`。vLLM servers 與佔位程式不變。

### 實作清單

gap check 報告的「B. 實作清單」第 1–41 項全部納入。缺一項即視為未完成，由稽核逐項核對。
