"""Tokenizer-only checks for the judge (no weights loaded).

1. Label judge (Llama-3.1-70B-Instruct): the fitted prompt encodes with exactly one BOS.
2. Trained judge (Qwen3-4B-Instruct-2507): training text = generation prompt + target, and the
   prompt token ids are an exact prefix of the training ids (train/inference parity), with the
   target ending in the template's end-of-turn token.
"""
import glob
import json

from transformers import AutoTokenizer

import goal_judge as GJ

LLAMA = glob.glob("/tmp2/hf_shared/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/*/")[0]
QWEN = glob.glob("/tmp2/hf_shared/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/")[0]


def main():
    hu, ha = ["I need plant image datasets with metadata"], ["Here are three datasets: A, B, C."]
    lt = AutoTokenizer.from_pretrained(LLAMA)
    msgs, info = GJ.fit_messages(lt, "persona and goal", hu, ha)
    text = lt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = lt(text, add_special_tokens=False)["input_ids"]
    bos = lt.bos_token_id
    assert ids.count(bos) == 1 and ids[0] == bos, ids[:5]
    assert lt(text)["input_ids"].count(bos) == 2, "sanity: default call would double the BOS"
    print("llama: one BOS ok; prompt tokens", info["prompt_tokens"])

    qt = AutoTokenizer.from_pretrained(QWEN)
    msgs, _ = GJ.fit_messages(qt, "persona and goal", hu, ha)
    prompt = qt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    target = GJ.canonical({"status": "PARTIAL", "unmet": ["metadata per sample"]})
    full = qt.apply_chat_template(msgs + [{"role": "assistant", "content": target}], tokenize=False,
                                  add_generation_prompt=False)
    assert full.startswith(prompt), (prompt[-80:], full[len(prompt) - 80:len(prompt) + 40])
    p_ids = qt(prompt, add_special_tokens=False)["input_ids"]
    f_ids = qt(full, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids
    tail = qt.decode(f_ids[len(p_ids):])
    assert tail.startswith(target) and "<|im_end|>" in tail, repr(tail)
    assert "<think>" not in full, "Instruct-2507 template must not inject think tags"
    print("qwen3-4b: train/inference prefix parity ok; target tail", repr(tail))
    print("JUDGE TOKENIZATION OK")


if __name__ == "__main__":
    main()
