"""Audit fixes (v16) in train_planner_rl.py: aux_floor_active, Task 1 group record for verify, end_prob under the GPU
lock, p_end of a valid decision without a stop mask, stale descriptions."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- stale descriptions
rep('''  4. advantages normalised once per group, split into the stop part (end_session tokens) and the rest;
     Task 1 stop groups on train_all conversations + the annealed stop supervision (D2);''',
    '''  4. group advantages R - mean(R) (v16 Dr. GRPO; no division by the group std), split into the stop part
     (end_session tokens) and the rest; Task 1 stop groups on train_all conversations (--task1-convs x 2 positions,
     --task1-G samples each, topped up with further conversations while groups carry no gradient) + the stop
     supervision (never below --stop-sup-floor; annealed towards it after the D2 trigger);''')
rep('''     -> validation.jsonl only; best.json = best checkpoint by w_sel_cov*coverage - w_sel_w1*W1(turn counts)
     + w_sel_task1*Task 1 term_f1 (M2); unclean validation episodes are re-run, else the score is withheld.''',
    '''     -> validation.jsonl only; best.json = best checkpoint by w_sel_cov*coverage - w_sel_w1*W1(turn counts)
     + w_sel_task1*Task 1 bal_p (v16: teacher-forced end probabilities at every decision point; term_f1 (M2) is
     reported); unclean validation episodes are re-run, else the score is withheld.''')
rep('''        self.aux_anneal_start = None # D2: first update at which validation Task 1 term_f1 beat task1_base''',
    '''        self.aux_anneal_start = None # D2 (v16): validation update at which bal_p beat task1_base by the margin twice in a row''')
rep('''            # of every decision step; coverage and the format constraints keep the sequence-level advantage
            # normalised once by the group's std of the TOTAL reward (not per part), so w_cov / w_dist and
            # the controller's factors keep their effect on the gradient''',
    '''            # of every decision step; coverage and the format constraints keep the sequence-level advantage;
            # both parts are centred on the TOTAL reward's group mean (v16: not divided by its std), so they add
            # up to the plain advantage and w_cov / w_dist and the controller's factors keep their effect''')
rep('''        # plus coverage; Task 1 by the M2 term_f1. The v4 validation reward is still logged (not selected on).
        # v16 (user 2026-09-28): Task 1 enters the selection through the continuous bal_p (20 decision points rather
        # than 4 end events); term_f1 stays the reported metric''',
    '''        # plus coverage. The v4 validation reward is still logged (not selected on).
        # v16 (user 2026-09-28): Task 1 enters the selection through the continuous bal_p (20 decision points rather
        # than 4 end events); term_f1 stays the reported metric''')
rep('''                    help="D2: updates over which the stop supervision goes linearly to 0 once validation Task 1 "
                         "term_f1 beats the untrained policy's; 0 = never anneal")''',
    '''                    help="D2: updates over which the stop supervision goes linearly towards --stop-sup-floor once "
                         "validation Task 1 bal_p beat the untrained policy's by --t1-trigger-margin at two consecutive "
                         "validations (v16); 0 = never anneal")''')
# ---- aux weight parts (floor active = the floor is above the annealed weight)
rep('''        validations, v16), then goes linearly to 0 over --stop-sup-anneal updates. v16 (user 2026-09-28): the
        supervision never goes below the floor."""
        w = float(cfg["w_aux"])
        if self.aux_anneal_start is not None and self.a.stop_sup_anneal > 0:
            w = w * max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))
        return max(float(self.a.stop_sup_floor), w)''',
    '''        validations, v16), then goes linearly to 0 over --stop-sup-anneal updates; the result never goes below
        the floor (v16, user 2026-09-28)."""
        return max(float(self.a.stop_sup_floor), self.aux_annealed(u, cfg))

    def aux_annealed(self, u, cfg):
        """cfg["w_aux"] x the D2 anneal, before the floor."""
        w = float(cfg["w_aux"])
        if self.aux_anneal_start is not None and self.a.stop_sup_anneal > 0:
            w = w * max(0.0, 1.0 - (u - self.aux_anneal_start) / float(self.a.stop_sup_anneal))
        return w''')
rep('''                "aux_floor_active": bool(w_aux > 0 and abs(w_aux - float(self.a.stop_sup_floor)) < 1e-12
                                         and float(cfg["w_aux"]) != w_aux),''',
    '''                "aux_annealed": self.aux_annealed(u, cfg),
                "aux_floor_active": bool(float(self.a.stop_sup_floor) > self.aux_annealed(u, cfg)),''')
# ---- the Task 1 groups of the update, for verify (rows of an aborted attempt / conversations without positions)
rep('''        self.t1_refill = {"n_base_groups": n_base, "n_refill_groups": len(out) - n_base, "n_refill_convs": n_refill_convs,
                          "n_informative_groups": sum(1 for r in out if informative(r)), "refill_stop": stop}''',
    '''        self.t1_refill = {"n_base_groups": n_base, "n_refill_groups": len(out) - n_base, "n_refill_convs": n_refill_convs,
                          "n_informative_groups": sum(1 for r in out if informative(r)), "refill_stop": stop,
                          "base_convs": sorted(cids),
                          "refill_convs": sorted({r["conversation_id"] for r in out if r.get("refill")}),
                          "groups": sorted([r["conversation_id"], r["t"], bool(r.get("refill"))] for r in out)}''')
# ---- end probability under the GPU lock; a valid decision without a stop mask counts as its greedy decision
rep('''                    pr = self.env.task1_end_probe(cid, x["t"], x["user_prompt"], x["real_final"])
                    pe = float(self.learner.end_prob(pr)) if pr["valid"] else 0.0''',
    '''                    pr = self.env.task1_end_probe(cid, x["t"], x["user_prompt"], x["real_final"])
                    if pr["valid"]:
                        # the learner's forward shares the GPU with the Speaker / Planner calls of Task2Env
                        with (getattr(self.env, "gpu_lock", None) or contextlib.nullcontext()):
                            pe = float(self.learner.end_prob(pr))
                    else:
                        # no value to score: an invalid plan reads as "not ending" (the benchmark's reading); a valid
                        # decision whose value tokens could not be located counts as its greedy decision
                        pe = 1.0 if (pr.get("decision_valid") and pr["greedy_end"]) else 0.0''')
rep('''import argparse
import copy
''', '''import argparse
import contextlib
import copy
''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
