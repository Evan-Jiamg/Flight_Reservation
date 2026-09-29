"""R6-1 fixes (en + zh)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
E = [
("The evaluation of the GRPO-trained Planner is ongoing. `TODO`: the GRPO results,",
 "Results of the GRPO-trained Planner are not reported in this version; they will be reported once all three folds have been trained and evaluated. `TODO`:"),
("The results reported here are those of the simulator before GRPO training (update 0); the evaluation of the GRPO-trained Planner is ongoing.",
 "The results reported here are those of the simulator before GRPO training (update 0); results of the GRPO-trained Planner are not reported in this version and will be reported once all three folds have been trained and evaluated."),
("Two systems are compared with a paired bootstrap over conversations (10,000 resamples).",
 "Two systems (for the GRPO comparison, to be reported) are compared with a paired bootstrap over conversations (10,000 resamples)."),
("with the credit for the length term given only to the end-session decision.",
 "with the credit for the length term given only to the end-session decision, plus an auxiliary supervised loss on the same stop decisions (weight floor 0.5)."),
]
Z = [
("GRPO 訓練後的 Planner 仍在評估中。`TODO`：GRPO 的結果、", "本版不報告 GRPO 訓練後 Planner 的結果；待三個 fold 都完成訓練與評估後再報告。`TODO`："),
("此處報告的是 GRPO 訓練前（update 0）的模擬器；GRPO 訓練後的 Planner 仍在評估中。",
 "此處報告的是 GRPO 訓練前（update 0）的模擬器；本版不報告 GRPO 訓練後 Planner 的結果，待三個 fold 都完成訓練與評估後再報告。"),
("兩個系統以對話為單位的 paired bootstrap（10,000 次重抽）比較。", "兩個系統（用於之後報告的 GRPO 比較）以對話為單位的 paired bootstrap（10,000 次重抽）比較。"),
("長度項的 credit 只歸給結束對話的決策。", "長度項的 credit 只歸給結束對話的決策，另加一個針對同一批結束決策的輔助監督損失（權重下限 0.5）。"),
]
for p, R in (("Draft.md", E), ("Draft_zh.md", Z)):
    s = open(p, encoding="utf-8").read()
    for a, b in R:
        assert s.count(a) == 1, (p, a[:50], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


def words(sec):
    return len(" ".join(l for l in sec.split("\n") if not l.startswith(">")).split())


s = open("Draft.md", encoding="utf-8").read()
c = [words(s.split("## 1. Introduction")[1].split("## 2. Related Work")[0]),
     words(s.split("## 2. Related Work")[1].split("**Condensed version")[0]),
     words(s.split("## 3. Methodology")[1].split("## 4. Future Work")[0]),
     words(s.split("## 4. Future Work")[1].split("## References")[0])]
print("words", c)
old = "本稿（rev.5）Intro 約 950 words、Related Work 約 570 words、Methodology 約 850 words、Future Work 約 800 words（皆不含作者註）"
new = "本稿（rev.6）Intro 約 %d words、Related Work 約 %d words、Methodology 約 %d words、Future Work 約 %d words（皆不含作者註）" % tuple(round(x, -1) for x in c)
for p in ("Draft.md", "Draft_zh.md"):
    t = open(p, encoding="utf-8").read()
    assert t.count(old) == 1, p
    open(p, "w", encoding="utf-8", newline="\n").write(t.replace(old, new))
print(new)
