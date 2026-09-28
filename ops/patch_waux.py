"""User decision 2026-09-25 (option B): the auxiliary stop-supervision weight w_aux is tuned by the v4 LLM
controller (not fixed). Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch("rl_controllers.py", [
    ('''TRAIN_DEFAULTS = {"lr": 1e-5, "kl_coef": 0.04}
TRAIN_BOUNDS = {"lr": (1e-7, 1e-4), "kl_coef": (0.0, 1.0)}''',
     '''# w_aux: weight of the auxiliary stop-token supervision on the Task 1 positions (D2 anneals it further);
# a training knob like lr / kl_coef, not part of the reward (so the shadow / selection reward ignore it)
TRAIN_DEFAULTS = {"lr": 1e-5, "kl_coef": 0.04, "w_aux": 1.0}
TRAIN_BOUNDS = {"lr": (1e-7, 1e-4), "kl_coef": (0.0, 1.0), "w_aux": (0.0, 10.0)}'''),
    ('''                "shadow_reward_mean", "turn_hist", "p_h")''',
     '''                "shadow_reward_mean", "turn_hist", "p_h", "aux_weight", "aux_stats", "task1_train")'''),
    ('''                 "keys": ["w_cov", "w_dist", "lambda_unparsed", "lambda_hit_max_new"],
                 "bounds": {"w_cov": [0.1, 5.0], "w_dist": [0.1, 5.0],
                            "lambda_unparsed": [0.1, 5.0], "lambda_hit_max_new": [0.1, 5.0]},''',
     '''                 "keys": ["w_cov", "w_dist", "lambda_unparsed", "lambda_hit_max_new", "w_aux"],
                 "bounds": {"w_cov": [0.1, 5.0], "w_dist": [0.1, 5.0],
                            "lambda_unparsed": [0.1, 5.0], "lambda_hit_max_new": [0.1, 5.0],
                            "w_aux": [0.01, 5.0]},'''),
    ('''    "- lambda_hit_max_new * (share of plans cut by the length cap). You see ONLY statistics of TRAINING "''',
     '''    "- lambda_hit_max_new * (share of plans cut by the length cap). Separately, w_aux weights an auxiliary "
    "supervised loss that pulls the Planner's end_session decision towards the real person's on training "
    "conversations (it is also annealed to 0 later); compare aux_grad_norm with grad_norm (the RL update) to "
    "judge whether it dominates or is negligible, and task1_train accuracy to judge whether it is still needed. "
    "You see ONLY statistics of TRAINING "'''),
    ('''                         "rate_unparsed": c.get("rate_unparsed"), "rate_hit_max_new": c.get("rate_hit_max_new"),
                         "kl": h.get("kl")})''',
     '''                         "rate_unparsed": c.get("rate_unparsed"), "rate_hit_max_new": c.get("rate_hit_max_new"),
                         "kl": h.get("kl"), "grad_norm": h.get("grad_norm"),
                         "aux_weight_effective": h.get("aux_weight"), "aux": h.get("aux_stats"),
                         "task1_train": {k: (h.get("task1_train") or {}).get(k)
                                         for k in ("acc", "end_at_final", "end_at_nonfinal")}})'''),
    ('''                lo, hi = self.opt["bounds"][k]
                cfg[k] = min(hi, max(lo, self.cfg[k] * f))
                applied[k] = {"factor": f, "value": cfg[k]}''',
     '''                lo, hi = self.opt["bounds"][k]
                if self.cfg[k] == 0:
                    applied[k] = {"factor": f, "value": 0.0, "note": "switched off by the run; stays 0"}
                    continue
                cfg[k] = min(hi, max(lo, self.cfg[k] * f))
                applied[k] = {"factor": f, "value": cfg[k]}'''),
])

patch("train_planner_rl.py", [
    ('''        cfg0 = RC.initial_cfg(**{**base_reward, **user_cfg.get("reward", {}), "lr": a.lr, "kl_coef": a.kl})''',
     '''        cfg0 = RC.initial_cfg(**{**base_reward, **user_cfg.get("reward", {}), "lr": a.lr, "kl_coef": a.kl,
                                 "w_aux": a.stop_sup_weight})'''),
    ('''    def aux_weight(self, u):
        """D2: full weight until validation Task 1 term_f1 first beats the untrained policy's, then linearly
        to 0 over --stop-sup-anneal updates."""
        w = self.a.stop_sup_weight''',
     '''    def aux_weight(self, u, cfg):
        """Effective stop-supervision weight = cfg["w_aux"] (initial --stop-sup-weight, then tuned by the LLM
        controller from TRAIN statistics) x the D2 anneal: 1 until validation Task 1 term_f1 first beats the
        untrained policy's, then linearly to 0 over --stop-sup-anneal updates."""
        w = float(cfg["w_aux"])'''),
    ('''        aux, w_aux = [], self.aux_weight(u)''',
     '''        aux, w_aux = [], self.aux_weight(u, cfg)'''),
    ('''                "aux_weight": w_aux, "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],''',
     '''                "aux_weight": w_aux, "lr": cfg["lr"], "kl_coef": cfg["kl_coef"],
                "aux_stats": {k: stats.get(k) for k in ("aux_n", "aux_loss", "aux_grad_norm", "aux_p_correct_before",
                                                         "aux_p_correct_end")},'''),
    ('''    ap.add_argument("--stop-sup-weight", type=float, default=1.0,''',
     '''    ap.add_argument("--stop-sup-weight", type=float, default=1.0,  # initial w_aux; the LLM controller tunes it'''),
])
print("ok")
