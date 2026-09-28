"""R5-2 optional nits (en + zh)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
E = [
("the visible number of requirements does not predict a user's real turn count (r = −0.36), and per-user counts vary little (s.d. 1.1).",
 "the visible number of requirements is only weakly (and negatively) related to a user's real turn count (r = −0.36; almost every user has 7 or 8 requirements), and per-user counts vary little (s.d. 1.1)."),
("、v11 run 的重選紀錄。", "。"),
("Two earlier training-free versions of this design showed why stopping needs this structure.",
 "Two earlier training-free versions of this design showed why stopping needs an explicit, binding decision."),
]
Z = [
("看得到的需求數無法預測使用者的真實輪數（r = −0.36），個人輪數的變化也很小（標準差 1.1）。",
 "看得到的需求數與使用者的真實輪數只有微弱的負相關（r = −0.36；幾乎每位使用者都有 7 或 8 項需求），個人輪數的變化也很小（標準差 1.1）。"),
("、v11 run 的重選紀錄。", "。"),
("這個設計先前的兩個免訓練版本說明了為什麼「結束」需要這樣的結構。",
 "這個設計先前的兩個免訓練版本說明了為什麼「結束」需要一個明確且具約束力的決策。"),
]
for p, R in (("Draft.md", E), ("Draft_zh.md", Z)):
    s = open(p, encoding="utf-8").read()
    for a, b in R:
        assert s.count(a) == 1, (p, a[:50], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
