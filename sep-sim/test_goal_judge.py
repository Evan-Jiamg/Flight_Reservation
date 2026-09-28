"""Pure-Python checks of the judge prompt builder, parser and status-prefix scoring plan (no model)."""
import math
import re

import goal_judge as J


class FakeTok:
    """Counts characters/4 as tokens; enough to exercise fitting logic."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join("<%s>%s" % (m["role"], m["content"]) for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "judge code must not add a second BOS"
        return {"input_ids": list(range(len(text) // 4))}


class BpeLikeTok:
    """Mimics a byte-level BPE pre-tokenizer (letters glued to one leading non-letter; punctuation
    runs glued; each pre-token split into 3-char pieces), so '", ' merges like in Qwen/Llama-3."""
    PAT = re.compile(r"[^\r\n\w]?[A-Za-z]+|\d| ?[^\s\w]+[\r\n]*|\s+")

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        ids = []
        for piece in self.PAT.findall(text):
            for k in range(0, len(piece), 3):
                ids.append(self.vocab.setdefault(piece[k:k + 3], len(self.vocab)))
        return {"input_ids": ids}


def test_status_probs_pure():
    tok = BpeLikeTok()
    prompt = "<system>judge<user>goal and dialogue<assistant>\n"
    shared, br, n_prompt = J.status_token_plan(tok, prompt)
    assert n_prompt == len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert shared == tok(prompt + J.STATUS_HEAD, add_special_tokens=False)["input_ids"][:len(shared)]
    assert set(br) == set(J.STATUSES) and all(br.values())
    # SATISFIED -> 3 pieces, PARTIAL -> 3, NOT -> 1; the '"S' / '"P' / '"N' starts already differ
    assert [len(br[s]) for s in J.STATUSES] == [3, 3, 1], br
    for st in J.STATUSES:       # scored ids == a prefix of the canonical training target ids
        full = tok(prompt + J.canonical({"status": st, "unmet": ["x"]}), add_special_tokens=False)["input_ids"]
        assert full[:len(shared) + len(br[st])] == shared + br[st]
    # why the closing quote is excluded: '{"status": "NOT"' is NOT a token prefix of the target
    pre_q = tok(prompt + '{"status": "NOT"', add_special_tokens=False)["input_ids"]
    full = tok(prompt + J.canonical({"status": "NOT", "unmet": []}), add_special_tokens=False)["input_ids"]
    assert full[:len(pre_q)] != pre_q

    # a tokenizer whose prefix does not match the target tokenization is refused, not scored
    class Glue(BpeLikeTok):
        def __call__(self, text, add_special_tokens=True):
            return super().__call__(text.replace('": "', '":"'), add_special_tokens)
    class Broken(BpeLikeTok):
        def __call__(self, text, add_special_tokens=True):
            if text.endswith(tuple(J.STATUSES)):
                text = text + "!"
            return super().__call__(text, add_special_tokens)
    J.status_token_plan(Glue(), prompt)
    try:
        J.status_token_plan(Broken(), prompt)
        raise AssertionError("expected a tokenization-parity failure")
    except ValueError:
        pass
    # normalization over the three prefixes
    p, mass = J.normalize_logps({"SATISFIED": math.log(0.2), "PARTIAL": math.log(0.1), "NOT": math.log(0.1)})
    assert abs(sum(p.values()) - 1) < 1e-12 and abs(p["SATISFIED"] - 0.5) < 1e-12 and abs(mass - 0.4) < 1e-12
    p, _ = J.normalize_logps({"SATISFIED": -1000.0, "PARTIAL": -1001.0, "NOT": -2000.0})   # no underflow
    assert abs(p["SATISFIED"] - 1 / (1 + math.exp(-1))) < 1e-12 and p["NOT"] == 0.0
    assert J.common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2 and J.common_prefix_len([[5], [6]]) == 0
    print("status_probs token plan / normalization ok")


def main():
    test_status_probs_pure()
    m = J.messages("persona  goal\n text", ["hi"], ["reply"])
    u = m[1]["content"]
    assert u.startswith("WHAT THIS PERSON WANTS\npersona goal text\n\nTHE CONVERSATION SO FAR\nUSER: hi\n\nASSISTANT: reply")
    for bad in ("turn", "MESSAGES", "useful replies", "pending"):
        assert bad not in u, bad
    assert "(nothing has been said yet)" in J.messages("g", [], [])[1]["content"]
    # parser
    ok = J.parse('Sure: {"status": "partial", "unmet": ["license", "  csv  format ", "", "x", "y"]}')
    assert ok == ({"status": "PARTIAL", "unmet": ["license", "csv format", "x"]}, True), ok
    assert J.parse('{"status": "SATISFIED", "unmet": ["leftover"]}')[0]["unmet"] == []
    assert J.parse("no json")[1] is False and J.parse("no json")[0]["status"] == "UNKNOWN"
    assert J.parse('{"status": "maybe"}')[1] is False
    assert J.parse('{"status": "NOT", "unmet": ["a",],}')[0] == {"status": "NOT", "unmet": ["a"]}
    assert J.canonical({"status": "NOT", "unmet": ["a"]}) == '{"status": "NOT", "unmet": ["a"]}'
    # fitting drops oldest exchanges only when needed
    hu = ["user %d " % i + "x" * 4000 for i in range(10)]
    ha = ["asst %d " % i + "y" * 4000 for i in range(10)]
    msgs, info = J.fit_messages(FakeTok(), "goal", hu, ha, budget=6000)
    assert info["dropped_exchanges"] > 0 and "user 9 " in msgs[1]["content"] and "user 0 " not in msgs[1]["content"]
    assert J.OMISSION in msgs[1]["content"]
    msgs, info = J.fit_messages(FakeTok(), "goal", hu[:1], ha[:1], budget=6000)
    assert info["dropped_exchanges"] == 0
    print("goal_judge builder/parser ok")


if __name__ == "__main__":
    main()
