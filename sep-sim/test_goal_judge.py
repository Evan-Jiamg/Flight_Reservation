"""Pure-Python checks of the judge prompt builder and parser (no model)."""
import goal_judge as J


class FakeTok:
    """Counts characters/4 as tokens; enough to exercise fitting logic."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join("<%s>%s" % (m["role"], m["content"]) for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "judge code must not add a second BOS"
        return {"input_ids": list(range(len(text) // 4))}


def main():
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
