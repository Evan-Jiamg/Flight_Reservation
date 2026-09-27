"""v16 items 6 (w_dist lower bound = its initial 1.0) and 4 (the controller prompt names the aux floor)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''                 "bounds": {"w_cov": [0.1, 5.0], "w_dist": [0.1, 5.0],''',
    '''                 # v16 (user 2026-09-28): w_dist may not go below its initial 1.0 (lowering it collapsed the
                 # conversation length in the v11 run, u12-u17); it may still be raised
                 "aux_floor": 0.0,
                 "bounds": {"w_cov": [0.1, 5.0], "w_dist": [1.0, 5.0],''')
rep('''"conversations (it is also annealed to 0 later); both share one optimizer step: compare aux_grad_norm with "''',
    '''"conversations (once validation Task 1 improves it is annealed towards a fixed floor of %g and never goes "
    "below it); both share one optimizer step: compare aux_grad_norm with "''')
rep('''                "messages": [{"role": "system", "content": LLM4_SYSTEM % (self.opt["factors"], ", ".join(self.opt["keys"]))},''',
    '''                "messages": [{"role": "system", "content": LLM4_SYSTEM % (float(self.opt["aux_floor"]), self.opt["factors"],
                                                                          ", ".join(self.opt["keys"]))},''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
