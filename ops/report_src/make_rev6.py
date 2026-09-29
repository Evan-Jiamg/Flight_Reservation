"""rev.6 (user 2026-09-29): GRPO is named in the Intro and fully described in the Methodology (the rev.4 text accepted
by the reviewer against the code); Results/Findings report only the simulator before GRPO training; §4 keeps the
training obstacles (incl. sample sizes) without the GRPO test comparison."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def between(s, a, b):
    i = s.index(a)
    return s[i:s.index(b, i)]


def sub(s, a, b, tag):
    assert s.count(a) == 1, (tag, a[:60], s.count(a))
    return s.replace(a, b)


# ------------------------------------------------------------------ English
s = open("Draft.md", encoding="utf-8").read()
r4 = open("rev4_en.md", encoding="utf-8").read()
s = sub(s, "(rev. 5)", "(rev. 6)", "en")
s = sub(s, "> 草稿 rev.5（2026-09-29）：依使用者決定重組——主方法是**免訓練**的 Planner–Speaker 模擬器；GRPO 移到 §4 Future Work，寫明遇到的難關（含樣本數）；Methodology 加上 PaperBanana 生成的架構圖（Figure 1）。",
        "> 草稿 rev.6（2026-09-29）：依使用者決定——Intro 直接點名 GRPO，Methodology 完整描述 GRPO（rev.4 已對照程式碼審過的 §3.3–3.4），Results／Findings 只放 **GRPO 訓練前**的結果（GRPO 版評估進行中）；§4 保留訓練難關（含樣本數），不放 GRPO 的 test 比較；Figure 1 由 PaperBanana 生成。", "en")
s = sub(s, "— a gap we first address by design (§3) and then try to close by training (§4).",
        "— a gap we address both by design and by session-level training (§3).", "en")
s = sub(s, "We also tried to learn stop timing with reinforcement learning (GRPO with session-level rewards); on held-out conversations it did not improve on the untrained Planner, and we report what made it hard (§4).",
        "On top of this design we train the Planner with GRPO on session-level rewards (§3.3): a term that moves the simulated distribution of session lengths towards the real one, LLM-judged requirement coverage, and agreement with real users' stop decisions, with the credit for the length term given only to the end-session decision.", "en")
s = sub(s, "> 註：Findings 目前只有 fold 2 的 test",
        "> 註：Findings 只放 **GRPO 訓練前**（update 0＝u0）的結果（使用者 2026-09-29 決定：Results 先不放 GRPO 版）；GRPO 版在 fold 2 test 的結果已跑出（`evidence_fw.txt`），暫不寫入。Findings 目前只有 fold 2 的 test", "en")
s = sub(s, "**Findings.** On the held-out test conversations of one fold (fold 2: five conversations for Task 2, each run with eight seeds, and nine for Task 1), the untrained simulator produces",
        "**Findings.** Before GRPO training, on the held-out test conversations of one fold (fold 2: five conversations for Task 2, each run with eight seeds, and nine for Task 1), the simulator produces", "en")
s = sub(s, "The GRPO-trained Planner did not improve on these numbers (§4). `TODO`: the other two folds, and the comparison with the earlier version under the same protocol.",
        "The evaluation of the GRPO-trained Planner is ongoing. `TODO`: the GRPO results, the other two folds, and the comparison with the earlier version under the same protocol.", "en")
s = sub(s, "- We design a training-free Planner–Speaker simulator in which",
        "- We design a Planner–Speaker simulator in which", "en")
s = sub(s, "on the held-out conversations of one fold its mean session length matches the real users' (5.65 vs 5.60 user messages; mean absolute difference 1.8 per episode) (`TODO`: all folds).",
        "before any training, on the held-out conversations of one fold, its mean session length matches the real users' (5.65 vs 5.60 user messages; mean absolute difference 1.8 per episode) (`TODO`: all folds).", "en")
i = s.index("- We report a negative result for session-level reinforcement learning:")
j = s.index("\n", i)
s = s[:i] + ("- We cast the stop decision as a session-level reinforcement-learning problem and train the Planner with GRPO: a reward "
             "that matches the real distribution of session lengths, agreement with real users' stop positions, and credit assignment "
             "to the end-session tokens (§3.3); we also report the practical obstacles to training it, starting with validation "
             "sets of one to four conversations (§4).") + s[j:]
s = sub(s, "> 5. 「免訓練」＝test 中的 u0：Qwen3-4B-Instruct-2507 加上一個**初始化為零效果**的 LoRA（等於原模型）。§4 的 RL 描述對應 AUDIT_SPEC / SPEC_v16；Dr. GRPO 需要引文（`TODO cite`）。",
        "> 5. 「GRPO 訓練前」＝test 中的 u0：Qwen3-4B-Instruct-2507 加上一個**初始化為零效果**的 LoRA（等於原模型）。§3.3–3.4 的 GRPO 描述對應 AUDIT_SPEC / SPEC_v16 與程式碼（rev.4 經 reviewer M1–M2 對照程式碼 ACCEPT）；Dr. GRPO 需要引文（`TODO cite`）。", "en")
s = sub(s, "only its use for a user whose stop decision is an explicit Planner output (and, in §4, the attempt to train it).",
        "only its use for a user whose stop decision is an explicit Planner output trained with session-level rewards (§3.3).", "en")
# §3 header note
s = sub(s, "> 註：本節只寫**已實作且在評估路徑上**的設計（`sep-sim/` 程式與 `ops/AUDIT_SPEC_pend_grpo.md`）。Figure 1 由 PaperBanana 依本節內容生成（見圖下註）。放進 2 頁模板時保留 Figure 1、3.2 前兩句與 3.3 的指標定義。",
        "> 註：本節只寫**已實作**的設計（`sep-sim/` 程式、`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）；§3.3–3.4 沿用 rev.4 經 reviewer 對照程式碼 ACCEPT 的文字。Figure 1 由 PaperBanana 依 §3.2 生成（見圖下註）。放進 2 頁模板時保留 Figure 1、3.2 前兩句、式 (1)(2) 與 3.5 的指標定義。", "en")
# §3.1: rev.4 paragraph (with the validation role) + keep the figure block of rev.5
old31 = between(s, "We follow the track's two tasks on the conversational data-search corpus.", "![Figure 1]")
new31 = between(r4, "We follow the track's two tasks on the conversational data-search corpus.", "### 3.2 Simulator")
s = sub(s, old31, new31, "en31")
s = sub(s, "**Planner.** At every turn a Planner (Qwen3-4B-Instruct-2507, used as is) reads",
        "**Planner.** At every turn a Planner (Qwen3-4B-Instruct-2507 with a LoRA adapter, the only trained component) reads", "en")
# §3.3–3.4 from rev.4, then §3.5 evaluation (from rev.5 §3.3, bal_p now refers to Eq. 5)
old33 = between(s, "### 3.3 Evaluation", "## 4. Future Work")
grpo = between(r4, "### 3.3 Training the Planner with GRPO", "> 註：\n> - 式 (1)–(6)")
grpo_notes = between(r4, "> 註：\n> - 式 (1)–(6)", "---\n\n## References")
ev = ("### 3.5 Evaluation\n\n"
      "We evaluate on the test split of each fold (so far fold 2). **Task 2**: each test scenario with requirement annotations "
      "(five in fold 2) is run with eight seeds (Planner temperature 0.7); we report the mean number of user messages against "
      "the real users', the Wasserstein-1 distance between the simulated and real turn counts (real counts capped at T_max), "
      "the mean absolute difference per episode, and the LLM-judged requirement coverage. **Task 1**: on every test "
      "conversation (nine in fold 2), the greedy run gives the stop decisions, scored as a stop F1 (our mapping) and as the "
      "fraction of conversations with a premature end, and the teacher-forced end probabilities give bal_p (Eq. 5) and the "
      "AUC of P_end between the real last message and the earlier ones. Two systems are compared with a paired bootstrap "
      "over conversations (10,000 resamples). The results reported here are those of the simulator before GRPO training "
      "(update 0); the evaluation of the GRPO-trained Planner is ongoing.\n\n"
      "> 註：評估程式 `sep-sim/eval_test_rl.py`（與訓練時 validate() 同一程序，只換成 test 的 id）與 `eval_test_boot.py`（重算每個數字＋paired bootstrap）；"
      "fold 2 結果在 `runs/pend_f2_v16/test_boot.txt`。Task 2 的 test 只取有需求標註的對話（coverage 需要），Task 1 用 test_all。"
      "coverage 對訓練前的系統不是 reward，可以當評估指標；對 GRPO 版它是 reward 的一項（D3），屆時要說明。\n\n")
s = sub(s, old33, grpo + grpo_notes + ev + "---\n\n", "en33")
# §4: obstacles without the GRPO test comparison
old4 = between(s, "## 4. Future Work: reinforcement learning for stop timing", "**Obstacles.**")
s = sub(s, old4, "## 4. Future Work: obstacles to training the stop decision\n\n"
        "> 註：依使用者決定，本節不放 GRPO 版的 test 比較，只寫訓練時遇到的難關與方向。數字來源：`evidence_fw*.txt`（fold 2 v16 run 的 validation／reselect、訓練 rollout 輪數、smoke 梯度、run_meta 時間）、"
        "`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`、`sep-sim/rl_controllers.py` 註解。\n\n"
        "Training the Planner with GRPO (§3.3) on this data raised practical obstacles that we are addressing.\n\n", "en4")
s = sub(s, "> - 原本完整的 GRPO 方法描述（含式、所有超參數與程式對應）保留在 git 歷史 rev.4（commit f14d4ef）與 `method_en.md`，需要時可放進附錄。\n", "", "en")
open("Draft.md", "w", encoding="utf-8", newline="\n").write(s)

# ------------------------------------------------------------------ Chinese
z = open("Draft_zh.md", encoding="utf-8").read()
r4z = open("rev4_zh.md", encoding="utf-8").read()
z = sub(z, "（rev. 5）", "（rev. 6）", "zh")
z = sub(z, "這是英文稿 `Draft.md` rev.5 的中文對照", "這是英文稿 `Draft.md` rev.6 的中文對照", "zh")
z = sub(z, "——我們先以設計處理這個缺口（§3），再嘗試以訓練補上它（§4）。", "——我們同時以設計與整段對話層級的訓練處理這個缺口（§3）。", "zh")
z = sub(z, "我們也嘗試用強化學習（以整段對話層級 reward 的 GRPO）學習結束時機；在未見過的對話上它沒有比未訓練的 Planner 更好，我們在 §4 報告遇到的難關。",
        "在這個設計之上，我們用 GRPO 以整段對話層級的 reward 訓練 Planner（§3.3）：一項把模擬的對話長度分佈推向真人分佈的 reward、由 LLM 判斷的需求涵蓋率，以及與真人結束決定的一致性；長度項的 credit 只歸給結束對話的決策。", "zh")
z = sub(z, "> 註：Findings 目前只有 fold 2 的 test",
        "> 註：Findings 只放 **GRPO 訓練前**（update 0＝u0）的結果（使用者 2026-09-29 決定：Results 先不放 GRPO 版）；GRPO 版在 fold 2 test 的結果已跑出（`evidence_fw.txt`），暫不寫入。Findings 目前只有 fold 2 的 test", "zh")
z = sub(z, "**主要發現。** 在一個 fold（fold 2：Task 2 有 5 段對話、每段跑 8 個 seed；Task 1 有 9 段對話）未見過的 test 對話上，未訓練的模擬器平均產生",
        "**主要發現。** 在 GRPO 訓練之前，於一個 fold（fold 2：Task 2 有 5 段對話、每段跑 8 個 seed；Task 1 有 9 段對話）未見過的 test 對話上，模擬器平均產生", "zh")
z = sub(z, "GRPO 訓練後的 Planner 沒有改善這些數字（§4）。`TODO`：另外兩個 fold，以及在相同協定下與前一版的比較。",
        "GRPO 訓練後的 Planner 仍在評估中。`TODO`：GRPO 的結果、另外兩個 fold，以及在相同協定下與前一版的比較。", "zh")
z = sub(z, "- 我們設計了一個免訓練的 Planner–Speaker 模擬器：", "- 我們設計了一個 Planner–Speaker 模擬器：", "zh")
z = sub(z, "在一個 fold 未見過的對話上，它的平均對話長度與真人相符",
        "在任何訓練之前，於一個 fold 未見過的對話上，它的平均對話長度與真人相符", "zh")
i = z.index("- 我們報告整段對話層級強化學習的負面結果：")
j = z.index("\n", i)
z = z[:i] + ("- 我們把結束決策表述為一個整段對話層級的強化學習問題，並以 GRPO 訓練 Planner：reward 對準真人的對話長度分佈、與真人結束位置的一致性，"
             "並把 credit 分配給結束對話的 token（§3.3）；我們也報告訓練它時遇到的實際難關，首先是只有 1 到 4 段對話的 validation（§4）。") + z[j:]
z = sub(z, "> 5. 「免訓練」＝test 中的 u0：Qwen3-4B-Instruct-2507 加上一個**初始化為零效果**的 LoRA（等於原模型）。§4 的 RL 描述對應 AUDIT_SPEC / SPEC_v16；Dr. GRPO 需要引文（`TODO cite`）。",
        "> 5. 「GRPO 訓練前」＝test 中的 u0：Qwen3-4B-Instruct-2507 加上一個**初始化為零效果**的 LoRA（等於原模型）。§3.3–3.4 的 GRPO 描述對應 AUDIT_SPEC / SPEC_v16 與程式碼（rev.4 經 reviewer M1–M2 對照程式碼 ACCEPT）；Dr. GRPO 需要引文（`TODO cite`）。", "zh")
z = sub(z, "讓使用者的結束決策成為 Planner 明確的輸出（並在 §4 嘗試訓練它）。",
        "讓使用者的結束決策成為 Planner 明確的輸出，並以整段對話層級的 reward 訓練它（§3.3）。", "zh")
z = sub(z, "> 註：本節只寫**已實作且在評估路徑上**的設計（`sep-sim/` 程式與 `ops/AUDIT_SPEC_pend_grpo.md`）。Figure 1 由 PaperBanana 依本節內容生成（見圖下註）。放進 2 頁模板時保留 Figure 1、3.2 前兩句與 3.3 的指標定義。",
        "> 註：本節只寫**已實作**的設計（`sep-sim/` 程式、`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）；§3.3–3.4 沿用 rev.4 經 reviewer 對照程式碼 ACCEPT 的文字。Figure 1 由 PaperBanana 依 §3.2 生成（見圖下註）。放進 2 頁模板時保留 Figure 1、3.2 前兩句、式 (1)(2) 與 3.5 的指標定義。", "zh")
old31 = between(z, "我們依照賽道在對話式資料集搜尋語料上的兩個任務。", "![Figure 1]")
new31 = between(r4z, "我們依照賽道在對話式資料集搜尋語料上的兩個任務。", "### 3.2 模擬器")
z = sub(z, old31, new31, "zh31")
z = sub(z, "每一輪，Planner（Qwen3-4B-Instruct-2507，直接使用原模型）讀取", "每一輪，Planner（Qwen3-4B-Instruct-2507 加上一個 LoRA adapter，是唯一被訓練的元件）讀取", "zh")
old33 = between(z, "### 3.3 評估", "## 4. 未來工作")
grpo = between(r4z, "### 3.3 以 GRPO 訓練 Planner", "> 註：\n> - 式 (1)–(6)")
grpo_notes = between(r4z, "> 註：\n> - 式 (1)–(6)", "---\n\n## 參考文獻")
evz = ("### 3.5 評估\n\n"
       "我們在每個 fold 的 test 上評估（目前是 fold 2）。**Task 2**：每個有需求標註的 test 情境（fold 2 有 5 個）各跑 8 個 seed（Planner temperature 0.7）；"
       "我們報告使用者訊息的平均數與真人的比較、模擬與真實輪數之間的 Wasserstein-1 距離（真實輪數以 T_max 為上限）、每個 episode 的平均絕對差，以及由 LLM 判斷的需求涵蓋率。"
       "**Task 1**：在每段 test 對話上（fold 2 有 9 段），greedy 執行給出結束決策，以結束 F1（我們的對應方式）與「有過早結束的對話比例」評分；"
       "teacher-forced 的結束機率則給出 bal_p（式 5）以及 P_end 在真實最後一則與較早各則之間的 AUC。兩個系統以對話為單位的 paired bootstrap（10,000 次重抽）比較。"
       "此處報告的是 GRPO 訓練前（update 0）的模擬器；GRPO 訓練後的 Planner 仍在評估中。\n\n"
       "> 註：評估程式 `sep-sim/eval_test_rl.py`（與訓練時 validate() 同一程序，只換成 test 的 id）與 `eval_test_boot.py`（重算每個數字＋paired bootstrap）；"
       "fold 2 結果在 `runs/pend_f2_v16/test_boot.txt`。Task 2 的 test 只取有需求標註的對話（coverage 需要），Task 1 用 test_all。"
       "coverage 對訓練前的系統不是 reward，可以當評估指標；對 GRPO 版它是 reward 的一項（D3），屆時要說明。\n\n")
z = sub(z, old33, grpo + grpo_notes + evz + "---\n\n", "zh33")
old4 = between(z, "## 4. 未來工作：以強化學習學習結束時機（Future Work）", "**難關。**")
z = sub(z, old4, "## 4. 未來工作：訓練結束決策的難關（Future Work）\n\n"
        "> 註：依使用者決定，本節不放 GRPO 版的 test 比較，只寫訓練時遇到的難關與方向。數字來源：`evidence_fw*.txt`（fold 2 v16 run 的 validation／reselect、訓練 rollout 輪數、smoke 梯度、run_meta 時間）、"
        "`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`、`sep-sim/rl_controllers.py` 註解。\n\n"
        "以 GRPO 訓練 Planner（§3.3）時，我們在這份資料上遇到以下實際難關，並正在處理。\n\n", "zh4")
z = sub(z, "> - 原本完整的 GRPO 方法描述（含式、所有超參數與程式對應）保留在 git 歷史 rev.4（commit f14d4ef）與 `method_zh.md`，需要時可放進附錄。\n", "", "zh")
open("Draft_zh.md", "w", encoding="utf-8", newline="\n").write(z)
print("ok")
