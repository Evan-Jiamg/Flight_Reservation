# -*- coding: utf-8 -*-
"""Planner, Speaker, Selector.

The split is the whole architecture. Every gradient goes to the planner; the
speaker is frozen and writes one hundred percent of the words. Loading is lazy
so the module imports, and the offline tests run, on a machine with no GPU.
"""
from __future__ import annotations

import os
import re

from . import acts, state

TAG_RE = re.compile(r"<\|[^|]*\|>")

#: Regeneration guards. The rule for what may live here is strict, because it
#: is the same rule as everywhere else in this package: a guard may test a
#: PROPERTY of a draw, never a list of words or a number someone arrived at by
#: looking at data.
#:
#: UserLM-8b's own paper (Naous et al., ICLR 2026, Appendix C.1) ships four
#: guardrails. Three of them fail that rule and are NOT used here:
#:
#:   1  First-token filter. Bans the logits of 'I', 'You' and 'Here' at the
#:      first position. Those three words were chosen by inspecting their own
#:      generations, so importing the list imports their fit. Rejected. The
#:      effect it was reaching for (repetitive openings) is a property, and a
#:      property is testable without naming any word.
#:   2  Termination suppression, which zeroes `<|endconversation|>`. We do the
#:      opposite and read that token as a signal, which is the point of the
#:      termination design.
#:   3  A 25-word cap and a 3-word floor. The cap is theirs for a reason that is
#:      theirs ('longer utterances tended to reveal the entire problem
#:      formulation'), and the floor came from inspection. Both rejected as
#:      magnitudes. An empty or contentless draw is still rejected, but on the
#:      property of being empty, not on a threshold.
#:   4  Verbatim-repetition filter: discard and draw again when the model
#:      repeats an earlier user turn or copies the intent word for word.
#:      Kept. It names no word and sets no threshold: repeating yourself
#:      exactly is a property of the draw.
#:
#: So exactly one of the four survives, and it survives because it is the only
#: one that is not a fit.
MAX_REGEN = 6

#: Where a rejected draw has to be redrawn but the draw that was rejected was
#: greedy. Re-running a deterministic decode returns the same words, so the
#: retry has to move off the mode. These are the sampler settings the arm is
#: already running for its own candidates, handed in by the caller; nothing
#: here is a number arrived at by looking at output.
RETRY_TEMPERATURE = 0.7
RETRY_TOP_P = 0.9

_WS = re.compile(r"\s+")
_MARKER_RE = None


def _norm(s):
    return _WS.sub(" ", (s or "")).strip().lower()


def template_marker_re():
    """Our own state block's field delimiters, read back out of the renderer.

    This is a property test and not a token list, and the difference is where
    the strings come from. Nobody read a generation to assemble these. The block
    is rendered once with sentinel values, every "- <name>:" label it emits is
    read off the result, and that set is the pattern. Rename a field or add one
    in `planner_prompt.render_block` or `state.render` and this moves with it,
    so the detector cannot drift away from what we actually emit.

    What it detects is therefore exactly one thing, and it is a thing we did:
    the speaker continued our scaffolding instead of speaking as the person.
    """
    global _MARKER_RE
    if _MARKER_RE is not None:
        return _MARKER_RE
    from . import planner_prompt  # local, so module import order stays free
    sent = "(sentinel)"
    probe = {k: sent for k in
             ("current_stage", "revealed", "unrevealed", "affect", "patience",
              "satisfaction", "terms", "move", "act", "bench_act", "next_step",
              "length_reason", "stop_rule")}
    probe["length_words"] = 99
    names = []
    for src in (planner_prompt.render_block(probe, (sent, sent)), state.render(probe)):
        for line in src.splitlines():
            if line.startswith("- ") and ":" in line:
                nm = line.split(":", 1)[0][2:].strip()
                if nm and nm not in names:
                    names.append(nm)
    if not names:                      # renderer changed shape; fail loud, not open
        raise RuntimeError("no field labels derived from the block renderer")
    _MARKER_RE = re.compile(
        r"(?:^|[\s\-*>|])(?:%s)\s*:" % "|".join(re.escape(n) for n in names), re.I)
    return _MARKER_RE


def looks_like_template(text):
    """Does the draw carry the delimiters of the block we handed the speaker?"""
    return bool(text) and bool(template_marker_re().search(text))


def is_verbatim_reuse(text, prior):
    """Did the draw reuse an the words of an earlier turn instead of writing new ones?

    The property is containment after case and whitespace normalisation, taken
    in both directions: the draw sits inside one of the session's earlier
    turns, or it swallows one whole. Equality -- guardrail 4 as Naous et al.
    ship it -- is the case where both hold at once, so this is their test with
    the incidental restriction to equal length removed. It stays a property:
    no word is named, no length is set, nothing was chosen by reading output.
    """
    a = _norm(text)
    if not a:
        return False
    for p in prior or []:
        b = _norm(p)
        if b and (a in b or b in a):
            return True
    return False


PLANNER_SYSTEM = """You maintain the running state of one specific person searching for a dataset.

Every USER message you are shown was typed by the real person. Their wording, punctuation and
length are evidence, not a simulation of it.

Each turn you do two things. First you criticise the state you wrote last turn against what has
actually happened since. Then you write the new state. You are NOT writing the person's message:
a separate model does that, and it sees only the state you write. Give it what it needs and
nothing it should not have.

THE MOVES. Choose one of the six, then one label from inside it:
%s

THE LABELS, by move:
%s

Decide the move BEFORE the label. The label must belong to the move you chose.

About one message in four is the last thing a person sends, and most do not read like an ending:
they ask a question at ordinary length and never come back. Judge whether this person has any
reason left to write again, not whether they have said goodbye. When they do not, the move is
Complete.

Reply with JSON only:
{"critique": "<one sentence on what your previous state got wrong, judged against what happened>",
 "voice": "<one short clause on how this person writes, drawn from their messages above>",
 "current_stage": "<exploring | refining | verifying | closing>",
 "revealed": "<what they have already said out loud, in their own words>",
 "unrevealed": "<what they still hold back>",
 "affect": "<their mood in a few words>",
 "patience": "<full | wearing | thin | spent>",
 "terms": "<two to five words or short phrases from THIS person's own request, as they wrote
           them, comma separated>",
 "move": "<Disclose, Reveal, Inquire, Navigate, Note, Complete, or Other>",
 "act": "<one label belonging to that move>",
 "next_step": "<one clause: what this turn does>"}""" % (acts.coarse_block(), acts.fine_block())


def build_planner_prompt(scenario_text, prev_block, hist_u, hist_a, turn, prev_ann=None):
    """The planner sees the previous block only. The speaker sees the current one.

    That asymmetry is the Lead's and it is deliberate: overwrite for the planner
    so its context does not grow without bound, accumulate for the speaker so the
    session reads as one person.
    """
    lines = []
    for i, u in enumerate(hist_u):
        lines.append("USER: " + u)
        if i < len(hist_a):
            lines.append("ASSISTANT: " + hist_a[i])
    convo = "\n\n".join(lines) if lines else "(they have not written anything yet)"
    obs = ""
    a = prev_ann or {}
    if a:
        bits = []
        if a.get("helpful") is not None:
            bits.append("They marked the assistant's last reply as %s."
                        % ("helpful" if a["helpful"] else "NOT helpful"))
        if a.get("dataset_quality"):
            bits.append("They rated what it offered: %s." % a["dataset_quality"])
        if a.get("feedback"):
            bits.append("Their note: %s" % a["feedback"])
        if bits:
            obs = ("\n\nWHAT THEY THOUGHT OF THE LAST REPLY (their own recorded judgement)\n"
                   + " ".join(bits))
    return ("WHO THEY ARE\n%s\n\nTHE STATE YOU WROTE LAST TURN\n%s\n\n"
            "THEY HAVE SENT %d MESSAGES SO FAR. You are writing the state for message %d."
            "%s\n\nTHE CONVERSATION SO FAR\n%s"
            % (scenario_text, prev_block, len(hist_u), turn, obs, convo))


class Planner:
    """Trainable. Writes a critique then a state block carrying a two-layer act."""

    def __init__(self, path=None, gpu=None, max_new=600):
        self.path = path or os.environ.get(
            "PLANNER_PATH",
            "/tmp2/TREC_UserSim_MingZhi/UserLM/00-models-v4GRPO-deps/Qwen2.5-32B-Instruct")
        self.gpu = gpu
        self.max_new = max_new
        self._tok = self._model = None
        self.n_unparsed = 0

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tok = AutoTokenizer.from_pretrained(self.path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map=({"": self.gpu} if self.gpu is not None else "auto")).eval()
        return self

    def raw(self, scenario_text, prev_block, hist_u, hist_a, turn, prev_ann=None):
        import torch
        if self._model is None:
            self.load()
        msgs = [{"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": build_planner_prompt(
                    scenario_text, prev_block, hist_u, hist_a, turn, prev_ann)}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt", truncation=True, max_length=8000)
        ids.pop("token_type_ids", None)
        if self.gpu is not None:
            ids = ids.to("cuda:%d" % self.gpu)
        with torch.no_grad():
            out = self._model.generate(
                **ids, max_new_tokens=self.max_new, do_sample=False,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        return self._tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def raw_with(self, system, user):
        """Generate against an arbitrary system/user pair.

        The v2 planner builds its own prompt from the persona priors, the ledger
        and the stopping rules, so it needs a way in that does not go through
        this class's own prompt construction.
        """
        import torch
        if self._model is None:
            self.load()
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt", truncation=True, max_length=12000)
        ids.pop("token_type_ids", None)
        if self.gpu is not None:
            ids = ids.to("cuda:%d" % self.gpu)
        with torch.no_grad():
            out = self._model.generate(
                **ids, max_new_tokens=self.max_new, do_sample=False,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        return self._tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def plan(self, scenario_text, prev_block, hist_u, hist_a, turn, prev_ann=None):
        """Returns (fields, block). Falls back to the previous block on a parse
        failure and counts it, because a silent reroll would hide the rate."""
        txt = self.raw(scenario_text, prev_block, hist_u, hist_a, turn, prev_ann)
        d = state.json_of(txt)
        if not d:
            self.n_unparsed += 1
            return None, prev_block
        p = state.from_plan(d, turn)
        # the platform's own annotations override the planner's guess where they exist
        if prev_ann:
            p.update(state.observed_state(prev_ann))
        return p, state.render(p)


class Speaker:
    """Frozen. Writes the whole utterance. Never receives a gradient."""

    def __init__(self, path=None, gpu=None, max_new=200, position="system",
                 guardrails=True):
        self.path = path or os.environ.get("SPEAKER_PATH", "/home/mzjiang/UserLM-8b")
        self.gpu = gpu
        self.max_new = max_new
        #: "system" or "prefix". D1 measured that an untrained UserLM continues a
        #: prefixed block as a document (55.4% marker leakage, 93 words against a
        #: human 31), so "prefix" is only correct after conditional SFT.
        self.position = position
        #: Naous et al.'s guardrails 1, 3-floor and 4. Guardrail 2 is inverted by
        #: design. Off switches them all, for the ablation that shows what they buy.
        self.guardrails = guardrails
        self._tok = self._model = None
        self.eot = None
        self.n_regen = 0
        self.n_reject_template = 0
        self.n_reject_reuse = 0

    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tok = AutoTokenizer.from_pretrained(self.path)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map=({"": self.gpu} if self.gpu is not None else "auto")).eval()
        self.eot = self._tok.convert_tokens_to_ids("<|eot_id|>")
        return self

    def build_prompt(self, scenario_text, block, hist_u, hist_a, turn):
        sys_msg = scenario_text
        if self.position == "system":
            sys_msg = scenario_text + "\n\n" + block
        msgs = [{"role": "system", "content": sys_msg}]
        for i, u in enumerate(hist_u):
            msgs.append({"role": "user", "content": u})
            if i < len(hist_a):
                msgs.append({"role": "assistant", "content": hist_a[i]})
        try:
            p = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            p = "".join("<|start_header_id|>%s<|end_header_id|>\n%s<|eot_id|>"
                        % (m["role"], m["content"]) for m in msgs)
            p += "<|start_header_id|>user<|end_header_id|>\n"
        if self.position == "prefix":
            p += '<profile turn="%d">\n%s\n</profile>\n' % (turn, block)
        return p

    def end_probability(self, scenario_text, hist_u, hist_a):
        """P(this turn is the end) read off the frozen speaker itself.

        UserLM-8b carries a dedicated `<|endconversation|>` token and scores 63.54
        F1 on dialogue termination where prompted assistants score 3 to 15
        (Naous et al., ICLR 2026, Table 5). Their own use of it is to SUPPRESS it:
        Appendix C.1 sets its probability to zero. Nobody in the 41 papers citing
        that work reads it as a signal instead.

        So we read it. One forward pass over the bare dialogue prefix gives the
        distribution over the first token of the next user turn, and the mass on
        `<|endconversation|>` is this model's own estimate that the person is
        done.

        The probe deliberately runs WITHOUT our state block. Conditioning it on a
        block whose `next_step` we wrote would make the planner's own intention
        come back to it as evidence. Bare, it is exogenous: a reading of the
        conversation, not of the plan.

        Nothing here is fitted. The number is the frozen model's, and the token is
        one the model was released with.
        """
        import torch
        if self._model is None:
            self.load()
        eoc = self._tok.convert_tokens_to_ids("<|endconversation|>")
        if eoc is None or eoc == self._tok.unk_token_id:
            return None
        msgs = [{"role": "system", "content": scenario_text}]
        for i, u in enumerate(hist_u):
            msgs.append({"role": "user", "content": u})
            if i < len(hist_a):
                msgs.append({"role": "assistant", "content": hist_a[i]})
        try:
            prompt = self._tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
        except Exception:
            prompt = "".join("<|start_header_id|>%s<|end_header_id|>" + chr(10) + "%s<|eot_id|>"
                             % (m["role"], m["content"]) for m in msgs)
            prompt += "<|start_header_id|>user<|end_header_id|>" + chr(10)
        ids = self._tok(prompt, return_tensors="pt", truncation=True, max_length=7000)
        ids.pop("token_type_ids", None)
        if self.gpu is not None:
            ids = ids.to("cuda:%d" % self.gpu)
        with torch.no_grad():
            logits = self._model(**ids).logits[0, -1]
        return float(torch.softmax(logits.float(), dim=-1)[eoc].item())

    def say(self, scenario_text, block, hist_u, hist_a, turn, seed=0,
            temperature=0.0, top_p=1.0, avoid=None,
            reject_template=False, reject_reuse=False,
            retry_temperature=RETRY_TEMPERATURE, retry_top_p=RETRY_TOP_P):
        """One utterance, with the model's own published guardrails applied.

        `avoid` is the session's earlier user turns. Guardrail 4 rejects a
        candidate that repeats one of them verbatim and draws again, up to a
        cap; on exhaustion the last draw is returned rather than looping, so a
        stubborn turn costs time but never hangs.

        Two further rejections are available and both are off unless asked for,
        so the arms generated before them reproduce unchanged:

        `reject_template`  the draw carries the field delimiters of the block we
                           handed it, i.e. the speaker continued the scaffolding
                           instead of speaking. The delimiters are derived from
                           the renderer, see `template_marker_re`.
        `reject_reuse`     the draw stands in a containment relation with an
                           earlier turn, which is guardrail 4's property without
                           the restriction to equal length.

        A rejected GREEDY draw is the case the shipped guard cannot handle: a
        deterministic decode redrawn is the same words, so the loop below has
        never been able to fire at temperature 0, and every arm so far ran its
        greedy draw unguarded. When a rejection is asked for, the retry moves
        off the mode using the caller's own sampler settings.
        """
        import torch
        if self._model is None:
            self.load()
        prompt = self.build_prompt(scenario_text, block, hist_u, hist_a, turn)
        ids = self._tok(prompt, return_tensors="pt", truncation=True, max_length=7000)
        ids.pop("token_type_ids", None)
        if self.gpu is not None:
            ids = ids.to("cuda:%d" % self.gpu)
        seen = [(x or "").strip().lower() for x in (avoid or []) if (x or "").strip()]
        prior = [x for x in (avoid or []) if (x or "").strip()]
        extra = bool(reject_template or reject_reuse)
        sampled = bool(temperature and temperature > 0)
        tries = MAX_REGEN if (self.guardrails and (sampled or extra)) else 1
        txt = ""
        ended = False
        for k in range(tries):
            torch.manual_seed(seed + 1000 * k)
            t_k, p_k = (temperature, top_p) if sampled else (
                (0.0, 1.0) if k == 0 else (retry_temperature, retry_top_p))
            kw = dict(max_new_tokens=self.max_new, eos_token_id=self.eot,
                      pad_token_id=self._tok.pad_token_id)
            kw.update(dict(do_sample=True, temperature=t_k, top_p=p_k)
                      if t_k and t_k > 0 else dict(do_sample=False))
            with torch.no_grad():
                out = self._model.generate(**ids, **kw)
            raw = self._tok.decode(out[0][ids["input_ids"].shape[1]:],
                                   skip_special_tokens=False)
            ended = "<|endconversation|>" in raw
            txt = re.sub(r"\s+", " ", TAG_RE.sub(" ", raw)).strip()
            if not self.guardrails:
                break
            low = txt.lower()
            # Properties, no thresholds: the draw said something, what it said
            # is not word for word something already said, it is not our own
            # scaffolding, and it is not contained in the context either way.
            if not low or low in seen:
                self.n_regen += 1
                continue
            if reject_template and looks_like_template(txt):
                self.n_reject_template += 1
                self.n_regen += 1
                continue
            if reject_reuse and is_verbatim_reuse(txt, prior):
                self.n_reject_reuse += 1
                self.n_regen += 1
                continue
            break
        return txt, ended


class DittoSpeaker(Speaker):
    """Ditto's user simulation interface with role-flipped dialogue history.

    Ditto is Qwen3-VL based and does not use UserLM's conversation-end token.
    A blank message is the benchmark's model-agnostic end signal.
    """

    def __init__(self, path=None, **kwargs):
        ditto_path = path or os.environ.get("DITTO_PATH")
        if not ditto_path:
            raise ValueError("DITTO_PATH must name a local Ditto-8B checkpoint")
        super().__init__(path=ditto_path, **kwargs)

    def load(self):
        import sys
        sys.modules["torchvision"] = None  # text-only Ditto; avoid mismatched optional torchvision
        sys.modules["torchaudio"] = None  # text-only Ditto; avoid mismatched optional torchaudio
        import torch
        from transformers import AutoTokenizer, AutoModelForImageTextToText

        self._tok = AutoTokenizer.from_pretrained(self.path)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map=({"": self.gpu} if self.gpu is not None else "auto")).eval()
        self.eot = self._tok.eos_token_id
        return self

    def build_prompt(self, scenario_text, block, hist_u, hist_a, turn):
        instruction = ("You are simulating a real human USER in a conversation "
                       "with an AI dataset-search assistant. Stay in character; "
                       "write ONLY the user's next message (no explanations, "
                       "no assistant-style answers).")
        system = instruction + "\n\n" + scenario_text + "\n\n" + block
        msgs = [{"role": "system", "content": system}]
        for i, utterance in enumerate(hist_u):
            msgs.append({"role": "assistant", "content": utterance})
            if i < len(hist_a):
                msgs.append({"role": "user", "content": hist_a[i]})
        try:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def say(self, *args, **kwargs):
        txt, _ = super().say(*args, **kwargs)
        return txt, not bool(txt.strip())

    def end_probability(self, scenario_text, hist_u, hist_a):
        return None


class StyleSelector:
    """Picks the candidate closest to this person's handwriting.

    No LLM judge. The benchmark protocol already pays for four candidates a turn
    and scores one; this is the reranker that is otherwise missing.
    """

    def __init__(self, path=None):
        self.path = path or os.environ.get(
            "STYLE_MODEL",
            "/tmp2/TREC_UserSim_MingZhi/models/princeton-nlp_sup-simcse-bert-base-uncased")
        self._enc = None

    def load(self):
        from sentence_transformers import SentenceTransformer
        self._enc = SentenceTransformer(self.path)
        return self

    def pick(self, candidates, reference_texts):
        """Index of the candidate whose embedding is nearest the person's own turns.

        Degrades to the first candidate rather than failing: the selector is an
        improvement on the protocol, not a dependency of it.
        """
        cands = [c for c in candidates if (c or "").strip()]
        if not cands:
            return 0
        if not reference_texts:
            return candidates.index(cands[0])
        try:
            if self._enc is None:
                self.load()
            import numpy as np
            C = self._enc.encode(cands, show_progress_bar=False)
            R = self._enc.encode(reference_texts, show_progress_bar=False).mean(0, keepdims=True)
            C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
            R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
            best = cands[int((C @ R.T).ravel().argmax())]
            return candidates.index(best)
        except Exception:
            return candidates.index(cands[0])
