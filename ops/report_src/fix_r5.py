"""Round R5-1 fixes (en + zh)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

E = [
# 9c gap bridge
("none compares a simulator's session lengths or stop positions with those of real users.",
 "none compares a simulator's session lengths or stop positions with those of real users — a gap we first address by design (§3) and then try to close by training (§4)."),
# 9b approach
("The current design grounds the decision in the Planner's own judgement of goal completion. We also tried",
 "The current design removes the rule-based stop ledger and lets the Planner's own `goal_met` / `still_wanted` judgement decide, and adds the implicit profile, the style-matched examples and the Borda selector; whether this improves on the second version has not yet been measured under the same protocol. We also tried"),
# 8 findings
("and in Task 1 never ends before the real user's last message (0 of 9 conversations), with a stop-decision F1 of 0.50 under our mapping and an end-probability AUC of 0.83.",
 "and in Task 1 never ends before the real user's last message (0 of 9 conversations) but ends at it in only 3 of 9 (stop F1 0.50 under our mapping; end-probability AUC 0.83)."),
# 8 C2
("on held-out conversations its session lengths are close to the real users' (`TODO`: all folds).",
 "on the held-out conversations of one fold its mean session length matches the real users' (5.65 vs 5.60 user messages; mean absolute difference 1.8 per episode) (`TODO`: all folds)."),
# 4 C3 hedge
("and an auxiliary supervision whose gradient dwarfs the RL gradient.",
 "and an auxiliary supervision whose gradient, in a smoke test, dwarfed the RL gradient."),
# 9a RW
("only its use for a user whose stop decision is trained.",
 "only its use for a user whose stop decision is an explicit Planner output (and, in §4, the attempt to train it)."),
# 11 RW note
("\n> - 空白的 Speaker 輸出也會結束 episode（AUDIT_SPEC），Method 節要寫清楚，避免讀者以為只有 Planner 能結束。", ""),
# 10 caption
("a selector picks one message, which is sent to the task agent. Illustration generated",
 "a selector picks one message, which is sent to the task agent (a closing message receives no reply; the Speaker also sees the conversation history). Illustration generated"),
("0 號 12 條連線與全部文字都符合規格；",
 "0 號 12 條連線與全部文字內容都符合規格（但小字如 \"same style, other users\"、\"4 candidates\"、Planner 欄位約只有規格要求字高 1/22 的一半，縮成單欄寬時可能太小，定稿前可考慮重生或放大）；"),
# 11 §3.3
("the mean absolute per-conversation difference, and the LLM-judged", "the mean absolute difference per episode, and the LLM-judged"),
# 7 what happened
("covered slightly fewer requirements (0.82 vs 0.86; [−0.08, 0.00]) and made worse stop decisions in Task 1 (stop F1 0.18 vs 0.50, [−0.62, 0.00]; one of nine conversations ended prematurely vs none).",
 "covered slightly fewer requirements (0.82 vs 0.86; [−0.08, +0.004]), and in Task 1 had a lower stop F1 (0.18 vs 0.50; [−0.62, 0.00]; one of nine conversations ended prematurely vs none) and lower but not separable bal_p and AUC; on seeds 0–1 alone, the seeds used for validation during training, the W1 difference points the other way (0.60 vs 1.00)."),
# 3 + 2 obstacles
("0.098, 0.213 and 0.095 — the ranking reversed. In an earlier run a checkpoint's score changed sign between two and eight seeds (0.281 → −0.176).",
 "0.098, 0.213 and 0.095 — the worst candidate under two seeds became the best under eight."),
# 4 aux
("In a smoke test the auxiliary stop loss produced a gradient about 130 times larger than the RL gradient (norm 0.040 vs 0.0003), so the direct supervision rather than the session-level reward drives the stop tokens.",
 "In a smoke update the auxiliary stop loss produced a gradient about 130 times the RL gradient (norm 0.040 vs 0.0003), which suggests that the supervision, not the session-level reward, drives the stop tokens."),
# 1 length
("- *Length collapse.* The length term can be satisfied by shortening every session: in an earlier run, lowering its weight collapsed the conversation length, and in the fold-2 run the mean length of training episodes fell from 6.4 to 3.5 user messages within ten updates while the real mean is 4.4–5.6.",
 "- *Length drift.* Other pressures shorten sessions faster than the length term restores them: in an earlier run, lowering the length weight collapsed the conversation length, and in the fold-2 run, although the weight could only rise, the mean length of training episodes fell from 6.4 to 3.5 user messages within ten updates, below the real mean (4.4 in the whole corpus, 5.6 in the fold-2 test conversations); we have not isolated the cause."),
# 6 hours
("ten updates took roughly eight to ten hours on two GPUs,", "ten updates with three validations took about eight hours on two GPUs,"),
# 5 PPO
("and compare GRPO with a value-based baseline (PPO), which our implementation already supports.",
 "and compare GRPO with a value-based baseline (PPO; our code has an untested PPO path that does not yet support the stop-token credit assignment)."),
# notes
("「0.281 → −0.176」出自 v11 run 的 u5 重選。", ""),
("「10 次 update 約 8–10 小時」是 v16 fold 2 的大約時間（05:29 開始；精確結束時間請以 run 紀錄為準）。",
 "「約 8 小時」：run_meta start 05:32，u10 的 validation summary 13:24（`evidence_fw3.txt`）。smoke 的梯度比只來自一次 Task-1-only smoke update；正式 run 每次 update 的 rl/aux 梯度範數有記錄但未納入證據。"),
("> - PPO 在 `rl_algos.py` 有實作（`--algo ppo`），但從未正式跑過。",
 "> - PPO 在 `rl_algos.py` 有實作（`--algo ppo`），但從未正式跑過；它需要 `--stop-credit 0` 與 `--ablation`，而且 `prepare()` 會用 value head 的 advantage 取代所有樣本（含 Task 1 結束樣本）的 advantage。"),
]

Z = [
("沒有任何一個拿模擬器的對話長度或結束位置去和真人比較。",
 "沒有任何一個拿模擬器的對話長度或結束位置去和真人比較——我們先以設計處理這個缺口（§3），再嘗試以訓練補上它（§4）。"),
("目前的設計讓 Planner 以自己對目標達成度的判斷作為結束的依據。我們也嘗試",
 "目前的設計拿掉了規則式的結束帳本，改由 Planner 自己的 `goal_met`／`still_wanted` 判斷決定，並加入隱含 profile、風格相符的範例與 Borda selector；這些改動是否比第二版更好，尚未在相同協定下量測。我們也嘗試"),
("在 Task 1 中從未在真人最後一則訊息之前結束（9 段中 0 段），以我們的對應方式計算的結束決策 F1 為 0.50，結束機率的 AUC 為 0.83。",
 "在 Task 1 中從未在真人最後一則訊息之前結束（9 段中 0 段），但只有 3 段在那一則結束（以我們的對應方式計算的結束 F1 為 0.50；結束機率的 AUC 為 0.83）。"),
("在未見過的對話上，它的對話長度接近真人（`TODO`：所有 fold）。",
 "在一個 fold 未見過的對話上，它的平均對話長度與真人相符（5.65 vs 5.60 則使用者訊息；每個 episode 的平均絕對差 1.8）（`TODO`：所有 fold）。"),
("以及輔助監督的梯度遠大於 RL 梯度。", "以及輔助監督的梯度在一次 smoke test 中遠大於 RL 梯度。"),
("新的只在於把它用在使用者端、並訓練使用者的結束決策。",
 "新的只在於把它用在使用者端，讓使用者的結束決策成為 Planner 明確的輸出（並在 §4 嘗試訓練它）。"),
("\n> - 空白的 Speaker 輸出也會結束 episode（AUDIT_SPEC），Method 節要寫清楚，避免讀者以為只有 Planner 能結束。", ""),
("selector 挑出一則訊息送給 task agent。插圖由", "selector 挑出一則訊息送給 task agent（收尾訊息不會得到回覆；Speaker 也看得到對話歷史）。插圖由"),
("0 號 12 條連線與全部文字都符合規格；",
 "0 號 12 條連線與全部文字內容都符合規格（但小字如 \"same style, other users\"、\"4 candidates\"、Planner 欄位約只有規格要求字高 1/22 的一半，縮成單欄寬時可能太小，定稿前可考慮重生或放大）；"),
("每段對話的平均絕對差，以及由 LLM 判斷的需求涵蓋率。", "每個 episode 的平均絕對差，以及由 LLM 判斷的需求涵蓋率。"),
("涵蓋的需求略少（0.82 vs 0.86；[−0.08, 0.00]），Task 1 的結束決策較差（結束 F1 0.18 vs 0.50，[−0.62, 0.00]；9 段中有 1 段過早結束，未訓練版為 0 段）。",
 "涵蓋的需求略少（0.82 vs 0.86；[−0.08, +0.004]）；Task 1 的結束 F1 較低（0.18 vs 0.50；[−0.62, 0.00]；9 段中有 1 段過早結束，未訓練版為 0 段），bal_p 與 AUC 也較低但無法區分；若只看訓練時 validation 用的 seeds 0–1，W1 的差異方向相反（0.60 vs 1.00）。"),
("用 8 個 seed 時是 0.098、0.213、0.095——排名反轉。在更早的一次 run 中，一個 checkpoint 的分數在 2 與 8 個 seed 之間甚至變號（0.281 → −0.176）。",
 "用 8 個 seed 時是 0.098、0.213、0.095——用 2 個 seed 時最差的候選，用 8 個 seed 時變成最好。"),
("在一次 smoke test 中，輔助結束損失的梯度約為 RL 梯度的 130 倍（範數 0.040 vs 0.0003），因此驅動結束 token 的是直接監督，而不是整段對話層級的 reward。",
 "在一次 smoke update 中，輔助結束損失的梯度約為 RL 梯度的 130 倍（範數 0.040 vs 0.0003），顯示驅動結束 token 的可能是直接監督，而不是整段對話層級的 reward。"),
("- *長度塌縮。* 長度項可以靠把每段對話縮短來滿足：在更早的一次 run 中，調低它的權重使對話長度塌縮；在 fold 2 的 run 中，訓練 episode 的平均長度在 10 次更新內從 6.4 降到 3.5 則使用者訊息，而真人平均是 4.4–5.6。",
 "- *長度漂移。* 其他壓力縮短對話的速度快過長度項把它拉回來的速度：在更早的一次 run 中，調低長度權重使對話長度塌縮；在 fold 2 的 run 中，儘管權重只能往上調，訓練 episode 的平均長度仍在 10 次更新內從 6.4 降到 3.5 則使用者訊息，低於真人平均（全語料 4.4、fold 2 test 的 5 段為 5.6）；原因我們尚未釐清。"),
("10 次更新在兩張 GPU 上約需 8 到 10 小時，", "10 次更新加上 3 次驗證在兩張 GPU 上約需 8 小時，"),
("並把 GRPO 與有 value baseline 的方法（PPO）比較——我們的實作已經支援 PPO。",
 "並把 GRPO 與有 value baseline 的方法（PPO）比較——我們的程式有一條未測試的 PPO 路徑，但它還不支援結束 token 的 credit 分配。"),
("「0.281 → −0.176」出自 v11 run 的 u5 重選。", ""),
("「10 次 update 約 8–10 小時」是 v16 fold 2 的大約時間（05:29 開始；精確結束時間請以 run 紀錄為準）。",
 "「約 8 小時」：run_meta start 05:32，u10 的 validation summary 13:24（`evidence_fw3.txt`）。smoke 的梯度比只來自一次 Task-1-only smoke update；正式 run 每次 update 的 rl/aux 梯度範數有記錄但未納入證據。"),
("> - PPO 在 `rl_algos.py` 有實作（`--algo ppo`），但從未正式跑過。",
 "> - PPO 在 `rl_algos.py` 有實作（`--algo ppo`），但從未正式跑過；它需要 `--stop-credit 0` 與 `--ablation`，而且 `prepare()` 會用 value head 的 advantage 取代所有樣本（含 Task 1 結束樣本）的 advantage。"),
]

for p, R in (("Draft.md", E), ("Draft_zh.md", Z)):
    s = open(p, encoding="utf-8").read()
    for a, b in R:
        assert s.count(a) == 1, (p, a[:60], s.count(a))
        s = s.replace(a, b)
    s = s.replace("。。", "。")
    open(p, "w", encoding="utf-8", newline="\n").write(s)


def words(sec):
    return len(" ".join(l for l in sec.split("\n") if not l.startswith(">")).split())


s = open("Draft.md", encoding="utf-8").read()
intro = s.split("## 1. Introduction")[1].split("## 2. Related Work")[0]
rw = s.split("## 2. Related Work")[1].split("**Condensed version")[0]
me = s.split("## 3. Methodology")[1].split("## 4. Future Work")[0]
fw = s.split("## 4. Future Work")[1].split("## References")[0]
print("intro", words(intro), "rw", words(rw), "method", words(me), "future", words(fw))
