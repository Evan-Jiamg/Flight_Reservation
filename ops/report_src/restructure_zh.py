"""rev.5 restructuring of the Chinese mirror (same content as restructure_en.py)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
p = "Draft_zh.md"
s = open(p, encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, (a[:50], s.count(a))
    s = s.replace(a, b)


def span(start, end_marker):
    i = s.index(start)
    return i, s.index(end_marker, i)


sub("Introduction、Related Work 與 Methodology（rev. 4）", "Introduction、Related Work、Methodology 與 Future Work（rev. 5）")
sub("這是英文稿 `Draft.md` rev.4 的中文對照", "這是英文稿 `Draft.md` rev.5 的中文對照")

i, j = span("**我們的做法。**", "> 註：下一段等 fold 2 test")
s = s[:i] + (
    "**我們的做法。** 我們建構一個 Planner–Speaker 模擬器，其中「結束對話」是 Planner（Qwen3-4B-Instruct-2507）一個獨立且具約束力的決策，"
    "由 Planner 自己判斷使用者的目標是否已達成；凍結的 Speaker（Ditto-8B，它沒有結束對話的 token；依 benchmark 慣例，Speaker 輸出空白也會結束 episode）"
    "根據計畫、Planner 在對話中累積的「這位使用者怎麼寫」的隱含 profile，以及寫作風格相同的其他使用者的真實訊息，寫出候選訊息，再由「長度＋風格」selector 挑出一則（Figure 1）。"
    "這個設計先前的兩個免訓練版本說明了為什麼「結束」需要這樣的結構。第一版中，Speaker 無視 Planner 的結束決定（被要求收尾時只有 8% 真的收尾），64 段中有 51 段撞到 10 輪上限。"
    "第二版在多項改動中包括讓 Planner 的決定具約束力，於是能夠收尾（K+1 end rate 0.05 → 0.48，對照的是去掉 annotations 的第一版），但結束得太早（premature end rate 0.03 → 0.14）。"
    "目前的設計讓 Planner 以自己對目標達成度的判斷作為結束的依據。我們也嘗試用強化學習（以整段對話層級 reward 的 GRPO）學習結束時機；"
    "在未見過的對話上它沒有比未訓練的 Planner 更好，我們在 §4 報告遇到的難關。\n\n") + s[j:]

i, j = span("> 註：下一段等 fold 2 test", "**貢獻。**")
s = s[:i] + (
    "> 註：Findings 目前只有 fold 2 的 test（Task 2：5 段對話 × 8 seeds；Task 1：9 段對話），fold 0/1 還沒跑；"
    "與前一版（E1.6）在同一協定下的比較也還沒做（AUDIT_SPEC 的比較協定）。若要與 benchmark 其他方法同表，須先證明我們 Task 2 環境"
    "（自架 R0／ledger judge）與 benchmark 的 system agent 設定（prompt v4、length_retry_v1）一致。數字來源："
    "`runs/pend_f2_v16/test_boot.txt`（u0 列；TEST CHECK PASSED、verify passed，2026-09-29 00:33）。\n\n"
    "**主要發現。** 在一個 fold（fold 2：Task 2 有 5 段對話、每段跑 8 個 seed；Task 1 有 9 段對話）未見過的 test 對話上，未訓練的模擬器平均產生 5.65 輪使用者訊息，"
    "真人為 5.60（輪數 W1 0.80）；涵蓋使用者 86% 的需求；在 Task 1 中從未在真人最後一則訊息之前結束（9 段中 0 段），"
    "以我們的對應方式計算的結束決策 F1 為 0.50，結束機率的 AUC 為 0.83。GRPO 訓練後的 Planner 沒有改善這些數字（§4）。"
    "`TODO`：另外兩個 fold，以及在相同協定下與前一版的比較。\n\n") + s[j:]

sub("- 我們讓「結束對話」成為 Planner 獨立且具約束力的決策，並以 GRPO 針對真人的對話長度與結束位置訓練它、把 credit 歸給結束決策，而不是用逐句 reward。\n- `TODO`（結果；在 goal 與 persona 都不重疊的 test sessions 上，於相同協定下與我們的免訓練版本比較。）",
    "- 我們設計了一個免訓練的 Planner–Speaker 模擬器：「結束對話」是明確且具約束力的決策，依據是 Planner 自己對目標達成度的判斷；Speaker 則以隱含 profile 與風格相符的真實訊息為條件。在未見過的對話上，它的對話長度接近真人（`TODO`：所有 fold）。\n"
    "- 我們報告整段對話層級強化學習的負面結果：以真人對話長度分佈與結束位置為 reward 的 GRPO，在未見過的對話上沒有贏過未訓練的 Planner；我們指出其中的難關——只有 1 到 4 段對話的 validation 讓 checkpoint 選擇充滿雜訊、結束訊號稀疏，以及輔助監督的梯度遠大於 RL 梯度。")

sub("> 5. 方法描述對應 AUDIT_SPEC / SPEC_v16：Task 1 stop groups 用 train_all 對話（不是 held-out）；validation 只用來挑 checkpoint（bal_p）；coverage 由 gpt-oss-120b ledger judge 判斷；Dr. GRPO 需要引文（`TODO cite`，來源裡沒有書目）。",
    "> 5. 「免訓練」＝test 中的 u0：Qwen3-4B-Instruct-2507 加上一個**初始化為零效果**的 LoRA（等於原模型）。§4 的 RL 描述對應 AUDIT_SPEC / SPEC_v16；Dr. GRPO 需要引文（`TODO cite`）。")

sub("> 註：本節只寫**已實作且在正式 run 路徑上**的設計（v16，`sep-sim/` 程式與 `ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）；每個數值都對應一個程式常數或 SPEC 值（見本節末的註）。放進 2 頁模板時保留 3.2 前兩句、式 (1)(2)(3) 與 3.4 的選擇分數，其餘移到附錄或全文版。",
    "> 註：本節只寫**已實作且在評估路徑上**的設計（`sep-sim/` 程式與 `ops/AUDIT_SPEC_pend_grpo.md`）。Figure 1 由 PaperBanana 依本節內容生成（見圖下註）。放進 2 頁模板時保留 Figure 1、3.2 前兩句與 3.3 的指標定義。")

i, j = span("我們依照賽道在對話式資料集搜尋語料上的兩個任務。", "### 3.2 模擬器")
s = s[:i] + (
    "我們依照賽道在對話式資料集搜尋語料上的兩個任務。**Task 2** 中，模擬器拿到 persona 與目標（主題、情境、使用者已知道的資料集），和一個 task agent 對話，"
    "直到它結束對話或達到 T_max = 10 則使用者訊息的上限。**Task 1** 中，模擬器以真實對話到第 t−1 則訊息為條件，產生第 t 則；同一次執行也得到它在每個真實回合"
    "「是否結束」的決定，我們把它當作結束決策來評分（這是我們自己的對應方式，不是官方的 Task 1 指標）。task agent 與評分涵蓋率的需求帳本（requirement ledger）"
    "都是本機的 gpt-oss-120b。我們採用 benchmark 的 goal 與 persona 都不重疊的三折切分：`train_all`（該 fold 的全部訓練對話）提供 Speaker 的 few-shot 範例，"
    "`test` 只讀一次，用於評估；`train` 與 `validation` 只用在 §4 的強化學習實驗。test 對話不會進入 few-shot 範例池。\n\n"
    "![Figure 1](fig/method_final.png)\n\n"
    "*Figure 1：Planner–Speaker 使用者模擬器。每一輪，Planner 寫出一份結構化計畫（包含是否結束對話的決定）；凍結的 Speaker 根據計畫、累積的 profile 筆記與風格相同的範例"
    "寫出候選訊息；selector 挑出一則訊息送給 task agent。插圖由 PaperBanana（`TODO` 模型名稱）依作者的方法描述生成，並經作者核對。*\n\n"
    "> 註：Figure 1 生成中／待使用者從候選中選定；圖中每條連線與文字需逐條核對（核對清單見 `fig/fig_method_spec.txt` 的 CONNECTIONS / FORBIDDEN）。"
    "若投稿場地有 AI 生圖政策，caption 的揭露句與 AI Declaration 都要保留。\n\n") + s[j:]

sub("每一輪，Planner（Qwen3-4B-Instruct-2507 加上一個 LoRA adapter，是唯一被訓練的元件）讀取", "每一輪，Planner（Qwen3-4B-Instruct-2507，直接使用原模型）讀取")

i, j = span("### 3.3 以 GRPO 訓練 Planner", "## 參考文獻")
s = s[:i] + (
    "### 3.3 評估\n\n"
    "我們在每個 fold 的 test 上評估（目前是 fold 2）。**Task 2**：每個有需求標註的 test 情境（fold 2 有 5 個）各跑 8 個 seed（Planner temperature 0.7）；"
    "我們報告使用者訊息的平均數與真人的比較、模擬與真實輪數之間的 Wasserstein-1 距離（真實輪數以 T_max 為上限）、每段對話的平均絕對差，以及由 LLM 判斷的需求涵蓋率。"
    "**Task 1**：在每段 test 對話上（fold 2 有 9 段），greedy 執行給出結束決策，以結束 F1（我們的對應方式）與「有過早結束的對話比例」評分；"
    "此外，在每個真實決策點 t，計算 Planner 結束對話的 teacher-forced 機率：以 greedy 計畫的前綴為條件，對 `end_session` 值計算 P_end = p(true) / (p(true) + p(false))，並以平衡分數\n\n"
    "  bal_p = ½ · mean_{t = n} P_end + ½ · mean_{2 ≤ t < n} (1 − P_end)  (1)\n\n"
    "（無效的點視為 P_end = 0）以及 P_end 在真實最後一則與較早各則之間的 AUC 來摘要。兩個系統以對話為單位的 paired bootstrap（10,000 次重抽）比較。\n\n"
    "> 註：評估程式 `sep-sim/eval_test_rl.py`（與訓練時 validate() 同一程序，只換成 test 的 id）與 `eval_test_boot.py`（重算每個數字＋paired bootstrap）；"
    "fold 2 結果在 `runs/pend_f2_v16/test_boot.txt`。Task 2 的 test 只取有需求標註的對話（coverage 需要），Task 1 用 test_all。"
    "coverage 對免訓練系統不是 reward 的一部分，所以可以當評估指標；但對 §4 的 RL 版它是 reward 的一項（D3）。\n\n"
    "---\n\n"
    "## 4. 未來工作：以強化學習學習結束時機（Future Work）\n\n"
    "> 註：本節內容取自原 §3.3–3.4（已經 reviewer 對照程式碼 ACCEPT），壓縮後加上結果與難關。數字來源：fold 2 v16 run（`runs/pend_f2_v16`：updates、validation、reselect、test_boot）、"
    "`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`、`sep-sim/rl_controllers.py` 註解、v11 run 的重選紀錄。\n\n"
    "**我們嘗試了什麼。** 近期的模擬器以逐句 reward 的 PPO 或 GRPO 訓練 [Abdulhai et al. 2025；Zhang et al. 2026；Wang et al. 2026a]。我們改以 GRPO 的一個變體，"
    "用**整段對話層級**的 reward 訓練 Planner（LoRA rank 16）。自由生成的 episode（每次更新 4 個訓練情境 × G = 4 個 episode）的 reward 是\n\n"
    "  R = w_cov · coverage + w_dist · [log p_h(T) − log q(T)] − Σ_k λ_k · rate_k ，  (2)\n\n"
    "其中 T 是使用者訊息數，p_h 是訓練 fold 中真人訊息數的平滑分佈，q 是本次更新 rollout 的同樣分佈（所以 E_q[log p_h − log q] = −KL(q ‖ p_h) 在長度的*分佈*與真人一致時最大），"
    "rate_k 是無法解析或被截斷的計畫比例。在 teacher-forced 的結束群組中（8 段訓練對話 × G₁ = 8 個樣本，取真實最後一則與一則較早的訊息），計畫的 `end_session` 與真人一致時 reward 為 1。"
    "Advantage 採用群組平均差、不除以標準差（Dr. GRPO，`TODO` 引文）：A_i = R_i − mean_j R_j；長度項造成的部分只歸給 `end_session` 值的 token。"
    "損失函數是 clipped surrogate，加上修正 vLLM 取樣的截斷重要性權重，以及對起始政策的 k3 KL 懲罰（β = 0.04），再加上一個對真人結束決策的輔助監督損失，其權重永不低於 0.5；"
    "一個 LLM 控制器每 5 次更新根據訓練統計重新調整 reward 權重。checkpoint 每 5 次更新在 validation 對話上驗證，以 coverage − W1(輪數) + bal_p 評分。\n\n"
    "**結果如何。** 在 fold 2，訓練在 10 次更新後停止（連續兩次驗證沒有進步）。以 8 個 seed 重新驗證 update 0、5、10 後選出 update 5（分數 0.213，另兩者為 0.098 與 0.095），"
    "但在 4 段 validation 對話上的 paired bootstrap 無法區分三者。在 test 對話上，update 5 的對話長度與未訓練的 Planner 相當（輪數 W1 0.83 vs 0.80；差值的 95% CI [−0.38, 0.65]），"
    "涵蓋的需求略少（0.82 vs 0.86；[−0.08, 0.00]），Task 1 的結束決策較差（結束 F1 0.18 vs 0.50，[−0.62, 0.00]；9 段中有 1 段過早結束，未訓練版為 0 段）。\n\n"
    "**難關。**\n"
    "- *validation 太小。* fold 2 的 validation 只有 4 段對話，fold 0 只有 1 段，所以 checkpoint 選擇被雜訊主導：用 2 個 seed 時三個候選的分數是 0.651、0.418、0.649，"
    "用 8 個 seed 時是 0.098、0.213、0.095——排名反轉。在更早的一次 run 中，一個 checkpoint 的分數在 2 與 8 個 seed 之間甚至變號（0.281 → −0.176）。\n"
    "- *test 太小。* 每個 fold 只有 5 段 Task 2 與 9 段 Task 1 對話，信賴區間比任何可預期的效果都寬；必須合併三個 fold 才能下結論。\n"
    "- *結束訊號稀疏。* 一段對話只有一個真實的結束點；在加入群組動態補抽之前，teacher-forced 結束群組中每 8 組有 5 到 8 組的 reward 完全相同（沒有梯度）。\n"
    "- *輔助監督壓過 RL。* 在一次 smoke test 中，輔助結束損失的梯度約為 RL 梯度的 130 倍（範數 0.040 vs 0.0003），因此驅動結束 token 的是直接監督，而不是整段對話層級的 reward。\n"
    "- *長度塌縮。* 長度項可以靠把每段對話縮短來滿足：在更早的一次 run 中，調低它的權重使對話長度塌縮；在 fold 2 的 run 中，訓練 episode 的平均長度在 10 次更新內從 6.4 降到 3.5 則使用者訊息，"
    "而真人平均是 4.4–5.6。\n"
    "- *無法預測的個人目標。* 以個人為單位的長度 reward 不可行：看得到的需求數無法預測使用者的真實輪數（r = −0.36），個人輪數的變化也很小（標準差 1.1）。\n"
    "- *成本。* 每次更新都需要完整的多輪 episode，外加一個 LLM task agent 與一個 LLM 判官；10 次更新在兩張 GPU 上約需 8 到 10 小時，限制了更新次數與 seed 數。\n\n"
    "**方向。** 跨 fold 合併 validation 與 test 對話（或使用重複交叉驗證），讓 checkpoint 選擇與評估有足夠的對話；給結束決策更密集的訊號（每段對話更多決策點，"
    "以及把整個預測的結束分佈與真實分佈比較的 reward）；讓輔助損失與 RL 梯度平衡，而不是固定它的下限；並把 GRPO 與有 value baseline 的方法（PPO）比較——我們的實作已經支援 PPO。\n\n"
    "> 註：\n"
    "> - 「130 倍」＝0.040 / 0.0003，出自 v16 smoke test（Task-1-only update）。「6.4 → 3.5」出自 fold 2 v16 的 updates.jsonl（訓練 rollout 平均輪數）；真人 4.4（benchmark 錨點，全語料）、5.6（fold 2 test 的 5 段）。"
    "「0.281 → −0.176」出自 v11 run 的 u5 重選。「r = −0.36、s.d. 1.1」出自 SPEC_v16 第 5 點。「5–8/8 組無梯度」出自 SPEC_v16 第 1 點（u1–u5）。"
    "「10 次 update 約 8–10 小時」是 v16 fold 2 的大約時間（05:29 開始；精確結束時間請以 run 紀錄為準）。\n"
    "> - PPO 在 `rl_algos.py` 有實作（`--algo ppo`），但從未正式跑過。\n"
    "> - 原本完整的 GRPO 方法描述（含式、所有超參數與程式對應）保留在 git 歷史 rev.4（commit f14d4ef）與 `method_zh.md`，需要時可放進附錄。\n\n"
    "---\n\n") + s[j:]

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
