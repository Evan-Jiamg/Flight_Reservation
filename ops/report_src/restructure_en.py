"""rev.5 restructuring (user 2026-09-29): the training-free Planner-Speaker simulator is the method; GRPO moves to a
Future Work section with the obstacles met; Methodology gets a figure (PaperBanana) and an evaluation subsection."""
import os
import re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
p = "Draft.md"
s = open(p, encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


def cut(start, end_marker):
    """Replace the text from `start` (inclusive) up to `end_marker` (exclusive)."""
    global s
    i = s.index(start)
    j = s.index(end_marker, i)
    return i, j


sub("# Draft — Introduction, Related Work & Methodology (rev. 4)",
    "# Draft — Introduction, Related Work, Methodology & Future Work (rev. 5)")
sub("> 草稿 rev.4（2026-09-28）：Intro／Related Work 經 reviewer 第 1–4 輪 ACCEPT；新增 §3 Methodology",
    "> 草稿 rev.5（2026-09-29）：依使用者決定重組——主方法是**免訓練**的 Planner–Speaker 模擬器；GRPO 移到 §4 Future Work，寫明遇到的難關（含樣本數）；Methodology 加上 PaperBanana 生成的架構圖（Figure 1）。")

# ---- Intro: approach, findings, contributions
i, j = cut("**Our approach.**", "> 註：下一段等 fold 2 test")
s = s[:i] + (
    "**Our approach.** We build a Planner–Speaker simulator in which ending the session is a separate, binding decision "
    "of a Planner (Qwen3-4B-Instruct-2507) that judges for itself whether the user's goal has been met; a frozen Speaker "
    "(Ditto-8B, which has no end-of-conversation token; a blank Speaker message also ends the episode, by the benchmark "
    "convention) writes candidate messages from the plan, from an implicit profile of how this user writes that the Planner "
    "accumulates over the conversation, and from real messages of users with the same writing style, and a length-and-style "
    "selector picks one (Figure 1). Two earlier training-free versions of this design showed why stopping needs this "
    "structure. In the first, the Speaker ignored the Planner's decision to stop (asked to close, it closed 8% of the time) "
    "and 51 of 64 sessions hit the 10-turn cap. A second version, which among other changes makes the Planner's decision "
    "binding, does close (K+1 end rate 0.05 → 0.48 against the first version without annotations) but ends too early "
    "(premature end rate 0.03 → 0.14). The current design grounds the decision in the Planner's own judgement of goal "
    "completion. We also tried to learn stop timing with reinforcement learning (GRPO with session-level rewards); on "
    "held-out conversations it did not improve on the untrained Planner, and we report what made it hard (§4).\n\n") + s[j:]

i, j = cut("> 註：下一段等 fold 2 test", "**Contributions.**")
s = s[:i] + (
    "> 註：Findings 目前只有 fold 2 的 test（Task 2：5 段對話 × 8 seeds；Task 1：9 段對話），fold 0/1 還沒跑；"
    "與前一版（E1.6）在同一協定下的比較也還沒做（AUDIT_SPEC 的比較協定）。若要與 benchmark 其他方法同表，須先證明我們 Task 2 環境"
    "（自架 R0／ledger judge）與 benchmark 的 system agent 設定（prompt v4、length_retry_v1）一致。數字來源："
    "`runs/pend_f2_v16/test_boot.txt`（u0 列；TEST CHECK PASSED、verify passed，2026-09-29 00:33）。\n\n"
    "**Findings.** On the held-out test conversations of one fold (fold 2: five conversations for Task 2, each run with "
    "eight seeds, and nine for Task 1), the untrained simulator produces sessions of 5.65 user turns on average against "
    "5.60 for the real users (turn-count W1 0.80), covers 86% of the users' requirements, and in Task 1 never ends before "
    "the real user's last message (0 of 9 conversations), with a stop-decision F1 of 0.50 under our mapping and an "
    "end-probability AUC of 0.83. The GRPO-trained Planner did not improve on these numbers (§4). `TODO`: the other two "
    "folds, and the comparison with the earlier version under the same protocol.\n\n") + s[j:]

sub("- We make ending the session a separate, binding Planner decision and train it with GRPO against real users' session lengths and stop positions, with credit assigned to the end-session decision, instead of per-utterance rewards.\n- `TODO` (result, on goal- and persona-disjoint test sessions, compared with our training-free version under the same protocol.)",
    "- We design a training-free Planner–Speaker simulator in which ending the session is an explicit, binding decision grounded in the Planner's own judgement of goal completion, with a Speaker conditioned on an implicit profile and on style-matched real messages; on held-out conversations its session lengths are close to the real users' (`TODO`: all folds).\n"
    "- We report a negative result for session-level reinforcement learning: GRPO with rewards on real users' session-length distribution and stop positions did not beat the untrained Planner on held-out conversations, and we identify the obstacles — validation sets of one to four conversations that make checkpoint selection noisy, a sparse stop signal, and an auxiliary supervision whose gradient dwarfs the RL gradient.")

sub("> 5. 方法描述對應 AUDIT_SPEC / SPEC_v16：Task 1 stop groups 用 train_all 對話（不是 held-out）；validation 只用來挑 checkpoint（bal_p）；coverage 由 gpt-oss-120b ledger judge 判斷；Dr. GRPO 需要引文（`TODO cite`，來源裡沒有書目）。",
    "> 5. 「免訓練」＝test 中的 u0：Qwen3-4B-Instruct-2507 加上一個**初始化為零效果**的 LoRA（等於原模型）。§4 的 RL 描述對應 AUDIT_SPEC / SPEC_v16；Dr. GRPO 需要引文（`TODO cite`）。")

# ---- §3 Methodology
sub("> 註：本節只寫**已實作且在正式 run 路徑上**的設計（v16，`sep-sim/` 程式與 `ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`）；每個數值都對應一個程式常數或 SPEC 值（見本節末的註）。放進 2 頁模板時保留 3.2 前兩句、式 (1)(2)(3) 與 3.4 的選擇分數，其餘移到附錄或全文版。",
    "> 註：本節只寫**已實作且在評估路徑上**的設計（`sep-sim/` 程式與 `ops/AUDIT_SPEC_pend_grpo.md`）。Figure 1 由 PaperBanana 依本節內容生成（見圖下註）。放進 2 頁模板時保留 Figure 1、3.2 前兩句與 3.3 的指標定義。")

i, j = cut("We follow the track's two tasks on the conversational data-search corpus.", "### 3.2 Simulator")
s = s[:i] + (
    "We follow the track's two tasks on the conversational data-search corpus. In **Task 2** the simulator receives a "
    "persona and a goal (topic, context, datasets the user already knows) and converses with a task agent until it ends "
    "the session or reaches the cap of T_max = 10 user messages. In **Task 1** it is conditioned on the real conversation "
    "up to message t−1 and produces message t; the same run also yields its decision to end at each real turn, which we "
    "score as a stop decision (our own mapping, not an official Task 1 measure). The task agent and the requirement ledger "
    "that scores coverage are a local gpt-oss-120b. We use the benchmark's goal- and persona-disjoint three-fold split: "
    "`train_all` (all training conversations of a fold) supplies the Speaker's few-shot examples, and `test` is read once, "
    "for the evaluation; `train` and `validation` are used only by the reinforcement-learning experiments of §4. No test "
    "conversation enters the few-shot pool.\n\n"
    "![Figure 1](fig/method_final.png)\n\n"
    "*Figure 1: The Planner–Speaker user simulator. At every user turn the Planner writes a structured plan, including the "
    "decision to end the session; a frozen Speaker writes candidate messages from the plan, the accumulated profile notes "
    "and same-style examples; a selector picks one message, which is sent to the task agent. Illustration generated with "
    "PaperBanana (`TODO` model name) from the authors' method description and verified by the authors.*\n\n"
    "> 註：Figure 1 生成中／待使用者從候選中選定；圖中每條連線與文字需逐條核對（核對清單見 `fig/fig_method_spec.txt` 的 CONNECTIONS / FORBIDDEN）。"
    "若投稿場地有 AI 生圖政策，caption 的揭露句與 AI Declaration 都要保留。\n\n") + s[j:]

sub("**Planner.** At every turn a Planner (Qwen3-4B-Instruct-2507 with a LoRA adapter, the only trained component) reads",
    "**Planner.** At every turn a Planner (Qwen3-4B-Instruct-2507, used as is) reads")

i, j = cut("### 3.3 Training the Planner with GRPO", "## References")
rl_block = s[i:j]
s = s[:i] + (
    "### 3.3 Evaluation\n\n"
    "We evaluate on the test split of each fold (so far fold 2). **Task 2**: each test scenario with requirement annotations "
    "(five in fold 2) is run with eight seeds (Planner temperature 0.7); we report the mean number of user messages against "
    "the real users', the Wasserstein-1 distance between the simulated and real turn counts (real counts capped at T_max), "
    "the mean absolute per-conversation difference, and the LLM-judged requirement coverage. **Task 1**: on every test "
    "conversation (nine in fold 2), the greedy run gives the stop decisions, scored as a stop F1 (our mapping) and as the "
    "fraction of conversations with a premature end; in addition, at every real decision point t we compute the "
    "teacher-forced probability that the Planner ends the session, P_end = p(true) / (p(true) + p(false)) for the "
    "`end_session` value given the greedy plan's prefix, and summarise it with the balanced score\n\n"
    "  bal_p = ½ · mean_{t = n} P_end + ½ · mean_{2 ≤ t < n} (1 − P_end)  (1)\n\n"
    "(invalid points count as P_end = 0) and with the AUC of P_end between the real last message and the earlier ones. "
    "Two systems are compared with a paired bootstrap over conversations (10,000 resamples).\n\n"
    "> 註：評估程式 `sep-sim/eval_test_rl.py`（與訓練時 validate() 同一程序，只換成 test 的 id）與 `eval_test_boot.py`（重算每個數字＋paired bootstrap）；"
    "fold 2 結果在 `runs/pend_f2_v16/test_boot.txt`。Task 2 的 test 只取有需求標註的對話（coverage 需要），Task 1 用 test_all。"
    "coverage 對免訓練系統不是 reward 的一部分，所以可以當評估指標；但對 §4 的 RL 版它是 reward 的一項（D3）。\n\n"
    "---\n\n"
    "## 4. Future Work: reinforcement learning for stop timing\n\n"
    "> 註：本節內容取自原 §3.3–3.4（已經 reviewer 對照程式碼 ACCEPT），壓縮後加上結果與難關。數字來源：fold 2 v16 run（`runs/pend_f2_v16`：updates、validation、reselect、test_boot）、"
    "`ops/SPEC_v16_grpo_opt.md`、`ops/AUDIT_SPEC_pend_grpo.md`、`sep-sim/rl_controllers.py` 註解、v11 run 的重選紀錄。\n\n"
    "**What we tried.** Recent simulators are trained with PPO or GRPO on per-utterance rewards [Abdulhai et al. 2025; "
    "Zhang et al. 2026; Wang et al. 2026a]. We instead trained the Planner (LoRA rank 16) with a GRPO variant on "
    "*session-level* rewards. For free-running episodes (four training scenarios × G = 4 episodes per update) the reward is\n\n"
    "  R = w_cov · coverage + w_dist · [log p_h(T) − log q(T)] − Σ_k λ_k · rate_k ,  (2)\n\n"
    "where T is the number of user messages, p_h the smoothed distribution of real users' message counts in the training "
    "fold, q the same distribution over the update's rollouts (so E_q[log p_h − log q] = −KL(q ‖ p_h) is maximised when the "
    "*distribution* of lengths matches the human one), and rate_k the rates of unparsable or truncated plans. In teacher-forced "
    "stop groups (eight training conversations × G₁ = 8 samples at the real last message and one earlier message) a plan "
    "gets reward 1 if its `end_session` agrees with the real user. Advantages are group-mean differences without "
    "standard-deviation normalisation (Dr. GRPO, `TODO` cite), A_i = R_i − mean_j R_j; the part caused by the length term is "
    "credited only to the `end_session` value tokens. The loss is the clipped surrogate with a truncated importance weight "
    "for vLLM sampling and a k3 KL penalty (β = 0.04) to the starting policy, plus an auxiliary supervised loss on the real "
    "users' stop decisions whose weight never falls below 0.5; an LLM controller retunes the reward weights every five "
    "updates from training statistics. Checkpoints are validated every five updates and scored by coverage − W1(turn counts) "
    "+ bal_p on the validation conversations.\n\n"
    "**What happened.** On fold 2, training stopped after ten updates (two validations without improvement). Re-validating "
    "update 0, 5 and 10 with eight seeds selected update 5 (score 0.213 against 0.098 and 0.095), but a paired bootstrap over "
    "the four validation conversations separated none of them. On the test conversations, update 5 matched the untrained "
    "Planner on session length (turn-count W1 0.83 vs 0.80; 95% CI of the difference [−0.38, 0.65]), covered slightly fewer "
    "requirements (0.82 vs 0.86; [−0.08, 0.00]) and made worse stop decisions in Task 1 (stop F1 0.18 vs 0.50, [−0.62, 0.00]; "
    "one of nine conversations ended prematurely vs none).\n\n"
    "**Obstacles.**\n"
    "- *Tiny validation sets.* The validation split has four conversations in fold 2 and one in fold 0, so checkpoint "
    "selection is dominated by noise: with two seeds the three candidates scored 0.651, 0.418 and 0.649, with eight seeds "
    "0.098, 0.213 and 0.095 — the ranking reversed. In an earlier run a checkpoint's score changed sign between two and eight "
    "seeds (0.281 → −0.176).\n"
    "- *Small test sets.* Five Task 2 and nine Task 1 conversations per fold give confidence intervals wider than any "
    "effect we could expect; the three folds must be pooled before any conclusion.\n"
    "- *Sparse stop signal.* A conversation has one real end; in the teacher-forced stop groups five to eight of every eight "
    "groups had identical rewards (no gradient) before we added dynamic re-sampling of groups.\n"
    "- *Auxiliary supervision dominates.* In a smoke test the auxiliary stop loss produced a gradient about 130 times larger "
    "than the RL gradient (norm 0.040 vs 0.0003), so the direct supervision rather than the session-level reward drives the "
    "stop tokens.\n"
    "- *Length collapse.* The length term can be satisfied by shortening every session: in an earlier run, lowering its weight "
    "collapsed the conversation length, and in the fold-2 run the mean length of training episodes fell from 6.4 to 3.5 user "
    "messages within ten updates while the real mean is 4.4–5.6.\n"
    "- *An unpredictable per-user target.* A per-user length reward is not viable: the visible number of requirements does "
    "not predict a user's real turn count (r = −0.36), and per-user counts vary little (s.d. 1.1).\n"
    "- *Cost.* Each update needs full multi-turn episodes with an LLM task agent and an LLM judge; ten updates took about "
    "eight hours on two GPUs, which limits the number of updates and seeds.\n\n"
    "**Directions.** Pool validation and test conversations across folds (or use repeated cross-validation) so that "
    "checkpoint selection and evaluation have enough conversations; give the stop decision a denser signal (more decision "
    "points per conversation, and rewards that compare the whole predicted stop distribution with the real one); balance "
    "the auxiliary loss against the RL gradient instead of fixing its floor; and compare GRPO with a value-based baseline "
    "(PPO), which our implementation already supports.\n\n"
    "> 註：\n"
    "> - 「130 倍」＝0.040 / 0.0003，出自 v16 smoke test（Task-1-only update）。「6.4 → 3.5」出自 fold 2 v16 的 updates.jsonl（訓練 rollout 平均輪數）；真人 4.4（benchmark 錨點，全語料）、5.6（fold 2 test 的 5 段）。"
    "「0.281 → −0.176」出自 v11 run 的 u5 重選。「r = −0.36、s.d. 1.1」出自 SPEC_v16 第 5 點。「5–8/8 組無梯度」出自 SPEC_v16 第 1 點（u1–u5）。「10 次 update 約 8 小時」是 v16 fold 2 的實際時間（05:29 開始）。\n"
    "> - PPO 在 `rl_algos.py` 有實作（`--algo ppo`），但從未正式跑過。\n"
    "> - 原本完整的 GRPO 方法描述（含式、所有超參數與程式對應）保留在 git 歷史 rev.4（commit f14d4ef）與 `method_en.md`，需要時可放進附錄。\n\n"
    "---\n\n") + s[j:]

open(p, "w", encoding="utf-8", newline="\n").write(s)
open("rl_block_rev4_en.md", "w", encoding="utf-8", newline="\n").write(rl_block)
print("ok")
