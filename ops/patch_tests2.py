p = "test_verify_pipeline.py"
s = open(p, encoding="utf-8").read()
old = '''        if ckpts:
            for u in ups:
                if isinstance(u.get("update"), int):
                    os.makedirs(os.path.join(rl, "ckpt", "u%05d" % u["update"]), exist_ok=True)
        return rl'''
new = '''        if ckpts:
            for u in ups:
                if isinstance(u.get("update"), int):
                    d = os.path.join(rl, "ckpt", "u%05d" % u["update"])
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, "rl_manifest.json"), "w") as f:
                        json.dump({"train_scenarios": ["tr1", "tr2"], "train_conversations": ["tr1", "tr2"],
                                   "fewshot_pool": []}, f)
        self.write("rl/validation.jsonl", [{"kind": "summary", "update": 1, "split": "validation"}], jsonl=True)
        self.write("rl/best.json", {"update": 1})
        return rl'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    def test_rl_missing_checkpoint(self):'''
new = '''    def test_rl_validation_on_train_id_and_manifest_leak(self):
        rl = self.make_rl()
        self.write("rl/validation.jsonl", [dict(wrapped(a2_episode("tr1"), 1, "validation"), kind="episode")], jsonl=True)
        with open(os.path.join(rl, "ckpt", "u00001", "rl_manifest.json"), "w") as f:
            json.dump({"train_scenarios": ["tr1", "va1"], "train_conversations": [], "fewshot_pool": []}, f)
        rc, rep, out = self.run_rl(rl)
        self.assertEqual((rc, rep.status("leak.validation_ids"), rep.status("leak.manifest")), (1, "FAIL", "FAIL"), out)

    def test_rl_missing_best(self):
        rl = self.make_rl()
        os.remove(os.path.join(rl, "best.json"))
        rc, rep, out = self.run_rl(rl)
        self.assertEqual((rc, rep.status("rl.best")), (1, "FAIL"), out)

    def test_rl_missing_checkpoint(self):'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
