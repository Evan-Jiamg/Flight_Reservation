"""Replace check_v16 in verify_pipeline.py with the audited version (ops/v16_check.py.txt)."""
import sys

p, newf = sys.argv[1], sys.argv[2]
s = open(p, encoding="utf-8").read()
i, j = s.index("def check_v16(rl_dir, rep):"), s.index("def check_rl(rl_dir, rollouts, ckpt_pattern, rep):")
new = open(newf, encoding="utf-8").read().rstrip() + "\n\n\n"
s = s[:i] + new + s[j:]
open(p, "w", encoding="utf-8").write(s)
print("replaced %d chars" % (j - i))
