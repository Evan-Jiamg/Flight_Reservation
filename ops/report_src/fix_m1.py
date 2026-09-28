"""Round M1 fixes for Draft.md / Draft_zh.md (Methodology review)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
E = [
("(Ditto-8B, whose own end token is masked; a blank Speaker message also ends the episode)",
 "(Ditto-8B, which has no end-of-conversation token; a blank Speaker message also ends the episode, by the benchmark convention)"),
("`validation` is used only to select checkpoints; `test` is read once, after training, for the final evaluation. No validation or test conversation enters training, the few-shot pool, the length distribution or the controller.",
 "`validation` selects checkpoints and, through one trigger, starts the annealing of the auxiliary stop loss (§3.3); `test` is read once, after training, for the final evaluation. No validation or test conversation is ever rolled out for a gradient, or enters the few-shot pool, the length distribution or the controller."),
("the user's last. The first message cannot end the session.",
 "the user's last. The message's act is drawn from this distribution; a closing (*Complete*) act drawn while `end_session` is false is redrawn from the other entries, so the message and the decision agree. The first message cannot end the session."),
("writes four candidate messages per turn (one greedy, three sampled at temperature 0.7, top-p 0.9) from the Planner's plan, the notes and k = 3 real messages of users with the same writing style, drawn from `train_all` of other goals and personas.",
 "writes four candidate messages per turn (one greedy and three sampled at temperature 0.7, top-p 0.9; at turn 1 all four are sampled at the checkpoint's own generation settings) from the Planner's plan, the notes and, for each candidate, its own k = 3 real messages of users with the same writing style, drawn from `train_all` conversations of other goals and personas."),
("are rejected, and a length-and-style Borda selector (SimCSE similarity to the user's own earlier messages, or to the examples) picks one. The Speaker's own end token is masked, so a session ends through the Planner's decision (or, by the benchmark convention, a blank message).",
 "are rejected (duplicates are redrawn, and if every candidate fails, extra candidates are drawn without examples), and a Borda selector ranks the survivors, with equal weight, by closeness to the Planner's target length and by SimCSE similarity to reference texts (the user's own earlier real messages in Task 1 from turn 2; otherwise the few-shot examples). Ditto-8B has no end-of-conversation token, so a session ends through the Planner's decision or, by the benchmark convention, when the selected Speaker message is blank."),
("Four training scenarios are drawn per update, and G = 4 episodes are rolled out for each (Planner temperature 1.0).",
 "Four training scenarios are drawn per update, and G = 4 episodes are rolled out for each (Planner temperature 1.0); the episodes of a group share the scenario seed (Speaker sampling, act draws, few-shot examples), so they differ only through the Planner's samples and the task agent."),
("At each point the Planner sees the real history and samples G₁ = 8 plans;",
 "At each point the Planner is given the prompt it reaches in a greedy teacher-forced pass of the current policy (the real history plus its own earlier state and notes) and samples G₁ = 8 plans at temperature 1;"),
("Its weight is w_eff = max(0.5, w_aux · a_u), where a_u = 1 until validation shows that the stop decisions have improved (the balanced end probability, §3.4, exceeds the untrained policy's by 0.10 at two consecutive validations), after which it decays linearly over 10 updates towards the floor, so the supervision never drops below 0.5.",
 "Its weight is w_eff = max(0.5, w_aux · a_u), where a_u = 1 until the balanced end probability (§3.4) is at least the untrained policy's + 0.10 at two consecutive validations, and then a_u = max(0.5, 1 − (u − u_trig)/10); w_aux starts at 1 and is tuned by the controller, so the supervision never drops below 0.5."),
("**Setup.** LoRA rank 16 (α = 32) on the attention projections, learning rate 2e-5, one optimizer step per update with gradient-norm clipping at 1.0;",
 "**Setup.** Initial weights w_cov = w_dist = λ_unparsed = λ_hit_max_new = 1. LoRA rank 16 (α = 32) on the attention projections; π_ref is the base model without the adapter (equal to the starting policy); AdamW (no weight decay), learning rate 2e-5, one optimizer step per update with gradient-norm clipping at 1.0;"),
("Every five updates the policy is validated on the fold's validation conversations:",
 "The untrained policy (update 0) and every fifth update are validated on the fold's validation conversations:"),
("which rewards ending at the real last message and continuing before it (invalid points count as P_end = 0).",
 "which rewards ending at the real last message and continuing before it (invalid points count as P_end = 0; a valid decision whose value tokens cannot be located counts as its greedy decision)."),
("and training stops after two validations without improvement (at most 30 updates). Because the validation split has only four conversations per fold, the candidate checkpoints are re-validated with eight seeds before one is chosen;",
 "and training stops after two validations without improvement (at most 30 updates). A validation with an episode that stays unclean after two re-runs gets no score. Because the validation splits are tiny (four conversations in fold 2, one in fold 0), the validated checkpoints are re-validated with eight seeds (0–7) and the best-scoring one is chosen (ties go to the earlier update);"),
("「8 seeds」是使用者 2026-09-27 核准的。",
 "「8 seeds」是使用者 2026-09-27 核准的。fold 2 已完成重選（u0/u5/u10 → u5）；fold 0/1 的腳本（`run_v16_fold.sh`）同樣對所有驗證過的 checkpoint 做 8-seed 重選。fold 0 的 validation 只有 1 段對話，如何挑 checkpoint 使用者尚未決定。"),
]
Z = [
("（Ditto-8B，它自己的 end token 被遮蔽；Speaker 輸出空白也會結束 episode）",
 "（Ditto-8B，它沒有結束對話的 token；依 benchmark 慣例，Speaker 輸出空白也會結束 episode）"),
("`validation` 只用來挑 checkpoint；`test` 在訓練結束後讀一次，做最終評估。validation 與 test 的對話都不會進入訓練、few-shot 範例池、長度分佈或控制器。",
 "`validation` 用來挑 checkpoint，並透過一個觸發條件啟動輔助結束損失的退火（§3.3）；`test` 在訓練結束後讀一次，做最終評估。validation 與 test 的對話從不用來跑產生梯度的 rollout，也不會進入 few-shot 範例池、長度分佈或控制器。"),
("若正在規劃的訊息是使用者的最後一則則為 true。第一則訊息不能結束對話。",
 "若正在規劃的訊息是使用者的最後一則則為 true。這則訊息的 act 從這份分佈抽出；若 `end_session` 為 false 卻抽到收尾（*Complete*）act，就改從其他項目重抽，讓訊息與決定一致。第一則訊息不能結束對話。"),
("凍結的 Speaker（Ditto-8B）每輪根據 Planner 的計畫、筆記，以及 k = 3 則寫作風格相同的使用者的真實訊息（取自 `train_all` 中其他 goal 與 persona 的對話），寫出四則候選訊息（一則 greedy、三則以 temperature 0.7、top-p 0.9 取樣）。",
 "凍結的 Speaker（Ditto-8B）每輪根據 Planner 的計畫、筆記，以及每則候選各自的 k = 3 則寫作風格相同的使用者真實訊息（取自 `train_all` 中其他 goal 與 persona 的對話），寫出四則候選訊息（一則 greedy、三則以 temperature 0.7、top-p 0.9 取樣；第 1 輪四則都以該 checkpoint 自己的生成設定取樣）。"),
("違反守衛（例如連續照抄範例 8 個字以上）的候選會被剔除，再由「長度＋風格」的 Borda selector（與使用者自己先前訊息、或與範例的 SimCSE 相似度）挑出一則。Speaker 自己的 end token 被遮蔽，所以對話只會透過 Planner 的決定（或依 benchmark 慣例，一則空白訊息）結束。",
 "違反守衛（例如連續照抄範例 8 個字以上）的候選會被剔除（重複的候選會重抽；若全部失敗，改以不帶範例的方式再抽），再由 Borda selector 以相同權重，依「與 Planner 目標長度的接近程度」與「與參考文本的 SimCSE 相似度」（Task 1 第 2 輪起用使用者自己先前的真實訊息，其餘情況用 few-shot 範例）為存活的候選排序。Ditto-8B 沒有結束對話的 token，所以對話只會透過 Planner 的決定、或依 benchmark 慣例在選出的 Speaker 訊息為空白時結束。"),
("每次更新抽 4 個訓練情境，每個情境跑 G = 4 個 episode（Planner temperature 1.0）。",
 "每次更新抽 4 個訓練情境，每個情境跑 G = 4 個 episode（Planner temperature 1.0）；同一群組的 episode 共用情境 seed（Speaker 取樣、act 抽取、few-shot 範例），所以彼此只因 Planner 的取樣與 task agent 而不同。"),
("在每個決策點，Planner 看到真實歷史並取樣 G₁ = 8 份計畫；",
 "在每個決策點，Planner 拿到目前政策在一次 greedy teacher-forced 執行中到達的 prompt（真實歷史，加上它自己先前的狀態與筆記），以 temperature 1 取樣 G₁ = 8 份計畫；"),
("其權重 w_eff = max(0.5, w_aux · a_u)：a_u 在驗證顯示結束決策已改善之前為 1（balanced end probability（§3.4）連續兩次驗證都比未訓練政策高出 0.10），之後在 10 次更新內線性遞減、趨向下限，所以監督權重永遠不低於 0.5。",
 "其權重 w_eff = max(0.5, w_aux · a_u)：在 balanced end probability（§3.4）連續兩次驗證都至少達到未訓練政策 + 0.10 之前，a_u = 1；之後 a_u = max(0.5, 1 − (u − u_trig)/10)。w_aux 初始為 1，並由控制器調整，所以監督權重永遠不低於 0.5。"),
("**設定。** LoRA rank 16（α = 32）加在 attention projection 上；learning rate 2e-5；每次更新做一次 optimizer step，梯度範數裁切於 1.0；",
 "**設定。** 初始權重 w_cov = w_dist = λ_unparsed = λ_hit_max_new = 1。LoRA rank 16（α = 32）加在 attention projection 上；π_ref 是不帶 adapter 的 base model（等於起始政策）；AdamW（無 weight decay）、learning rate 2e-5；每次更新做一次 optimizer step，梯度範數裁切於 1.0；"),
("每 5 次更新，在該 fold 的 validation 對話上驗證一次政策：",
 "未訓練的政策（update 0）與每第 5 次更新，會在該 fold 的 validation 對話上驗證："),
("它獎勵在真實最後一則結束、在那之前繼續（無效的點視為 P_end = 0）。",
 "它獎勵在真實最後一則結束、在那之前繼續（無效的點視為 P_end = 0；決策有效但找不到值 token 的點，以它的 greedy 決定計）。"),
("連續兩次驗證沒有進步就停止訓練（最多 30 次更新）。由於每個 fold 的 validation 只有 4 段對話，候選 checkpoint 會先以 8 個 seed 重新驗證再選定；",
 "連續兩次驗證沒有進步就停止訓練（最多 30 次更新）。若某次驗證有 episode 重跑兩次後仍不乾淨，該次不給分數。由於 validation 很小（fold 2 有 4 段對話，fold 0 只有 1 段），驗證過的 checkpoint 會以 8 個 seed（0–7）重新驗證，選分數最高者（平手取較早的 update）；"),
("「8 seeds」是使用者 2026-09-27 核准的。",
 "「8 seeds」是使用者 2026-09-27 核准的。fold 2 已完成重選（u0/u5/u10 → u5）；fold 0/1 的腳本（`run_v16_fold.sh`）同樣對所有驗證過的 checkpoint 做 8-seed 重選。fold 0 的 validation 只有 1 段對話，如何挑 checkpoint 使用者尚未決定。"),
]
for p, R in (("Draft.md", E), ("Draft_zh.md", Z)):
    s = open(p, encoding="utf-8").read()
    for a, b in R:
        assert s.count(a) == 1, (p, a[:50], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
