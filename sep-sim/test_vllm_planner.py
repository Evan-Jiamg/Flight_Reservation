# -*- coding: utf-8 -*-
"""VLLMPlanner against a local fake vLLM server: exact ids, behaviour log-probs, cap detection, and every
inconsistency raising instead of being repaired; adapter naming / swapping; the TIS weight formula."""
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import rl_algos as RA
import vllm_planner as VP

EOS = 151645


class State:
    mode = "stop"
    served = ["planner-base"]
    loaded = []
    unloaded = []
    last_body = None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.endswith("/v1/models"):
            self._send({"data": [{"id": m, "max_model_len": 2048} for m in State.served]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        State.last_body = body
        if self.path.endswith("/load_lora_adapter"):
            State.served.append(body["lora_name"])
            State.loaded.append(body["lora_name"])
            return self._send({"message": "ok"})
        if self.path.endswith("/unload_lora_adapter"):
            State.served.remove(body["lora_name"])
            State.unloaded.append(body["lora_name"])
            return self._send({"message": "ok"})
        m = State.mode
        ids = [5, 6, EOS] if m in ("stop", "no_ids", "bad_lp") else ([5, 6, 7] if m == "length" else [5, 6])
        ch = {"text": "x", "finish_reason": "length" if m == "length" else "stop",
              "logprobs": {"token_logprobs": [-0.1, -0.2] if m == "bad_lp" else [-0.1] * len(ids)}}
        if m != "no_ids":
            ch["token_ids"] = ids
        self._send({"choices": [ch]})


class Tok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not (skip_special_tokens and i == EOS))


class FakePlanner:
    max_new, budget, n_calls = 3, 1000, 0
    tok = Tok()

    def __init__(self, n_prompt=10):
        self.n_prompt = n_prompt

    def build_prompt(self, it):
        return "user", {"compacted": False}, list(range(self.n_prompt))

    def eos_ids(self):
        return {EOS, 151643}


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield "http://127.0.0.1:%d/v1" % srv.server_address[1]
    srv.shutdown()


ITEM = {"system": "s", "user": "u", "temperature": 1.0, "top_p": 1.0, "seed": 7}


def test_generation_fields_exact(server):
    State.mode = "stop"
    v = VP.VLLMPlanner(server)
    p = FakePlanner()
    g = v.generate_batch(p, [ITEM])[0]
    assert g["prompt_ids"] == list(range(10)) and g["gen_ids"] == [5, 6, EOS]
    assert g["gen_logprobs"] == [-0.1, -0.1, -0.1] and g["hit_max_new"] is False
    assert g["gen_adapter"] == "planner-base" and g["fit"]["backend"] == "vllm" and g["fit"]["prompt_tokens"] == 10
    b = State.last_body
    assert b["prompt"] == list(range(10)) and b["max_tokens"] == 3 and b["return_token_ids"] and b["logprobs"] == 1
    assert b["seed"] == 7
    greedy = dict(ITEM, temperature=0.0, top_p=0.9)
    v.generate_batch(p, [greedy])
    assert State.last_body["temperature"] == 0.0 and State.last_body["top_p"] == 1.0 and "seed" not in State.last_body


def test_cap_is_detected(server):
    State.mode = "length"
    g = VP.VLLMPlanner(server).generate_batch(FakePlanner(), [ITEM])[0]
    assert g["hit_max_new"] is True and g["gen_ids"] == [5, 6, 7]


@pytest.mark.parametrize("mode,msg", [("no_ids", "token_ids"), ("bad_lp", "log-probs"), ("stop_no_eos", "end token")])
def test_inconsistencies_raise(server, mode, msg):
    State.mode = mode
    with pytest.raises(RuntimeError, match=msg):
        VP.VLLMPlanner(server).generate_batch(FakePlanner(), [ITEM])


def test_prompt_beyond_server_context_raises(server):
    State.mode = "stop"
    with pytest.raises(AssertionError, match="max_model_len"):
        VP.VLLMPlanner(server).generate_batch(FakePlanner(n_prompt=2046), [ITEM])


def test_adapter_names_carry_version_and_sha_and_old_ones_are_unloaded(server, tmp_path):
    State.mode = "stop"
    v = VP.VLLMPlanner(server)
    a0, a1 = tmp_path / "u0", tmp_path / "u1"
    for d, w in ((a0, b"zero"), (a1, b"one")):
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(w)
    n0 = v.use_adapter(str(a0), "p0")
    assert n0.startswith("p0-") and n0 == "p0-" + VP.sha_dir(str(a0))[:12]
    assert v.use_adapter(str(a0), "p0") == n0 and State.loaded.count(n0) == 1        # same policy: no reload
    n1 = v.use_adapter(str(a1), "p1")
    assert n1 != n0 and n0 in State.unloaded and v.current == n1
    assert VP.VLLMPlanner(server).generate_batch(FakePlanner(), [ITEM])[0]["gen_adapter"] == "planner-base"
    assert v.generate_batch(FakePlanner(), [ITEM])[0]["gen_adapter"] == n1


def test_tis_weights():
    w = RA.tis_weights([-1.0, -0.5, -2.0], [-1.0, -2.0, -0.5], 2.0)
    assert w[0] == 1.0 and w[1] == 2.0 and math.isclose(w[2], math.exp(-1.5))
