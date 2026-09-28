"""Round-4 audit fix (GRPO audit #1): the aux loss joins the LAST GRPO minibatch's backward and optimizer
step (one step per update, clipped together), so w_aux is a real relative weight under Adam. Run from sep-sim/."""


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="\n").write(s)


patch("rl_algos.py", [
    ('''        tok_seen = 0
        for ep in range(int(self.acfg["epochs"])):
            for mb in minibatches(len(samples), int(self.acfg["minibatches"]), seed * 1000 + ep):
                n_tok = sum(len(samples[i]["gen_ids"]) for i in mb)''',
     '''        tok_seen = 0
        if aux and any("gen_len" not in x for x in aux):
            raise ValueError("aux example without gen_len: cannot normalise like the GRPO loss")

        def aux_backward():
            """Auxiliary stop-token supervision: -w * log p(the human's end_session value | prompt + the policy's
            own prefix), normalised like the GRPO loss by the GENERATED tokens of the generations the values
            belong to. Its gradient is ADDED to the current .grad (the RL gradient of the last minibatch), so
            RL and aux share one clipped optimizer step; its own norm is measured for the controller."""
            before = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            n_t = sum(int(x["gen_len"]) for x in aux)
            p_before = []
            for x in aux:
                lp, _ = token_logprobs(self.model, list(x["prompt_ids"]) + list(x["prefix_ids"]),
                                       x["target_ids"], 1.0)
                p_before.append(float(lp.detach().sum().exp()))
                loss = -float(x["weight"]) * lp.sum() / n_t
                loss.backward()
                st["aux_loss"] = st.get("aux_loss", 0.0) + float(loss.detach())
            sq = 0.0
            for p, b in zip(params, before):
                if p.grad is not None:
                    d = p.grad.detach().float() - (b.float() if b is not None else 0.0)
                    sq += float((d ** 2).sum())
            st["aux_grad_norm"] = sq ** 0.5
            st["aux_n"] = len(aux)
            st["aux_p_correct_before"] = sum(p_before) / len(p_before)
            st["aux_p_correct_end"] = (sum(p for p, x in zip(p_before, aux) if x["want_end"]) /
                                       max(1, sum(1 for x in aux if x["want_end"])))

        n_ep = int(self.acfg["epochs"])
        for ep in range(n_ep):
            mbs = list(minibatches(len(samples), int(self.acfg["minibatches"]), seed * 1000 + ep))
            for k_mb, mb in enumerate(mbs):
                last_mb = ep == n_ep - 1 and k_mb == len(mbs) - 1
                n_tok = sum(len(samples[i]["gen_ids"]) for i in mb)'''),
    ('''                gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
                st["grad_norm"].append(float(gn))
                self.optimizer.step()
                st["optimizer_steps"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        if aux:
            # auxiliary stop-token supervision (after the on-policy GRPO steps, so it never touches the
            # ratio check): -log p(the human's end_session value | prompt + the policy's own prefix)
            _set_mode(self.model, self.acfg["forward_mode"])
            # normalised like the GRPO loss: by the number of GENERATED tokens of the generations these values
            # belong to (not by the 1-2 target tokens, which made the aux gradient ~1400x the RL gradient in the
            # v8 smoke); w_aux is then a relative weight, tuned by the LLM controller
            if any("gen_len" not in x for x in aux):
                raise ValueError("aux example without gen_len: cannot normalise like the GRPO loss")
            n_t = sum(int(x["gen_len"]) for x in aux)
            p_before = []
            for x in aux:
                lp, _ = token_logprobs(self.model, list(x["prompt_ids"]) + list(x["prefix_ids"]),
                                       x["target_ids"], 1.0)
                p_before.append(float(lp.detach().sum().exp()))
                loss = -float(x["weight"]) * lp.sum() / n_t
                loss.backward()
                st["aux_loss"] = st.get("aux_loss", 0.0) + float(loss.detach())
            gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
            st["aux_grad_norm"] = float(gn)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            st["optimizer_steps"] += 1
            st["aux_n"] = len(aux)
            st["aux_p_correct_before"] = sum(p_before) / len(p_before)
            st["aux_p_correct_end"] = (sum(p for p, x in zip(p_before, aux) if x["want_end"]) /
                                       max(1, sum(1 for x in aux if x["want_end"])))''',
     '''                if last_mb:
                    st["rl_grad_norm"] = sum(float((p.grad.detach().float() ** 2).sum())
                                             for p in params if p.grad is not None) ** 0.5
                    if aux:
                        aux_backward()        # same backward pass, same clip, same optimizer step
                gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
                st["grad_norm"].append(float(gn))
                self.optimizer.step()
                st["optimizer_steps"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        if aux and not samples:
            # an update with supervision only (no RL sample survived): one step for the aux loss alone
            _set_mode(self.model, self.acfg["forward_mode"])
            st["rl_grad_norm"] = 0.0
            aux_backward()
            gn = torch.nn.utils.clip_grad_norm_(params, self.acfg["max_grad_norm"])
            st["grad_norm"].append(float(gn))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            st["optimizer_steps"] += 1'''),
])

patch("rl_controllers.py", [
    ('''                "shadow_reward_mean", "turn_hist", "p_h", "aux_weight", "aux_stats", "task1_train")''',
     '''                "shadow_reward_mean", "turn_hist", "p_h", "aux_weight", "aux_stats", "task1_train", "rl_grad_norm")'''),
    ('''                         "kl": h.get("kl"), "grad_norm": h.get("grad_norm"),''',
     '''                         "kl": h.get("kl"), "rl_grad_norm": h.get("rl_grad_norm"),'''),
    ('''    "conversations (it is also annealed to 0 later); compare aux_grad_norm with grad_norm (the RL update) to "''',
     '''    "conversations (it is also annealed to 0 later); both share one optimizer step: compare aux_grad_norm with "
    "rl_grad_norm (the RL part of the same step) to "'''),
])

patch("train_planner_rl.py", [
    ('''                **{k: stats.get(k) for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                              "value_mse", "ratio_init_maxdev")}}''',
     '''                **{k: stats.get(k) for k in ("loss", "kl", "ratio_mean", "clip_frac", "grad_norm", "n_tokens",
                                              "value_mse", "ratio_init_maxdev", "rl_grad_norm")}}'''),
    # FakeLearner mirrors the real learner: aux normalised by gen_len, no extra optimizer step with RL samples
    ('''    def _aux_step(self, aux, cfg, st):
        lr = cfg["lr"] * self.lr_scale
        ps = []
        for x in aux:
            k = x["prompt_ids"][0]
            p = _sig(self.theta[k])
            ps.append(p if x["target_ids"][0] == 1 else 1 - p)
            grad = (1 - p) if x["target_ids"][0] == 1 else -p
            self.theta[k] += lr * float(x["weight"]) * grad / len(aux)
        st["aux_n"], st["aux_p_correct_before"] = len(aux), sum(ps) / len(ps)
        st["optimizer_steps"] = st.get("optimizer_steps", 0) + 1''',
     '''    def _aux_step(self, aux, cfg, st, own_step=False):
        lr = cfg["lr"] * self.lr_scale
        n_t = sum(int(x["gen_len"]) for x in aux)            # normalised per generated token, like the real learner
        ps = []
        for x in aux:
            k = x["prompt_ids"][0]
            p = _sig(self.theta[k])
            ps.append(p if x["target_ids"][0] == 1 else 1 - p)
            grad = (1 - p) if x["target_ids"][0] == 1 else -p
            self.theta[k] += lr * float(x["weight"]) * grad / n_t
        st["aux_n"], st["aux_p_correct_before"] = len(aux), sum(ps) / len(ps)
        if own_step:                                           # aux-only update: its own step
            st["optimizer_steps"] = st.get("optimizer_steps", 0) + 1'''),
    ('''            self._aux_step(aux, cfg, out)
            return out''',
     '''            self._aux_step(aux, cfg, out, own_step=True)
            return out'''),
])
print("ok")
