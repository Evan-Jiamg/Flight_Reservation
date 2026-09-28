"""Audit P fixes in train_planner_rl.py: F6 (the floor acts on the anneal factor, so the controller's w_aux keeps its
effect after the anneal - as told to the user), F1 (SPEC gate for earlier approved values), F5 (probe adapter)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''    def aux_annealed(self, u, cfg):
        """cfg["w_aux"] x the D2 anneal, before the floor."""
        w = float(cfg["w_aux"])
        if self.aux_anneal_start is not None and self.a.stop_sup_anneal > 0:
            w = w * max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))
        return w''', '''    def aux_annealed(self, u, cfg):
        """cfg["w_aux"] x the D2 anneal factor, before the floor. The factor goes linearly from 1 towards
        floor / --stop-sup-weight (not to 0): once the anneal is over the controller's w_aux still scales the
        supervision (user 2026-09-28: the floor bounds it from below, the controller may still raise it)."""
        w = float(cfg["w_aux"])
        if self.aux_anneal_start is not None and self.a.stop_sup_anneal > 0:
            fmin = (float(self.a.stop_sup_floor) / float(self.a.stop_sup_weight)) if self.a.stop_sup_weight > 0 else 0.0
            f = max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))
            w = w * max(fmin, f)
        return w''')
# F1: earlier approved values are spec values too (outside the dry run for the sizes the dry-run tests shrink)
rep('''    off = {k: getattr(a, k) for k in SPEC if getattr(a, k) != SPEC[k]}
    if a.controller != "llm":''', '''    off = {k: getattr(a, k) for k in SPEC if getattr(a, k) != SPEC[k]}
    for k, want in (("G", 4), ("behav_mismatch_abort", 0.1), ("w_sel_cov", 1.0), ("w_sel_w1", 1.0),
                    ("w_sel_task1", 1.0), ("batch", 1), ("task1_tol", 0.05)):
        if getattr(a, k) != want:
            off[k] = getattr(a, k)
    if a.stop_sup_weight not in (0.0, 1.0):
        off["stop_sup_weight"] = a.stop_sup_weight
    if not a.dry_run:
        for k, want in (("scenarios_per_update", 4), ("val_every", 5)):
            if getattr(a, k) != want:
                off[k] = getattr(a, k)
    if a.controller != "llm":''')
# F5: the adapter that produced a validation probe's greedy prefix
rep('''            probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                          "decision_valid": bool(pr.get("decision_valid", pr["valid"])),
                          "greedy_end": bool(pr["greedy_end"])})''',
    '''            probs.append({"t": x["t"], "real_final": bool(x["real_final"]), "p_end": pe, "valid": bool(pr["valid"]),
                          "decision_valid": bool(pr.get("decision_valid", pr["valid"])),
                          "greedy_end": bool(pr["greedy_end"]), "gen_adapter": pr.get("gen_adapter")})''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
