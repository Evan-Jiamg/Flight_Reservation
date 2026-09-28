# -*- coding: utf-8 -*-
"""make_floor_judge: floor, one re-request through a separate cache, per-episode incident counters."""
import os

import pytest

import task2_env as T2


class FakeJudge:
    answers = []            # consumed in order, shared by the main judge and its retry judge
    calls = []

    def __init__(self, reasoning_effort="minimal", verbose=False, cache_dir=None, model="gpt-oss-120b"):
        self.model, self.cache_dir = model, cache_dir

    def chat(self, system, user, max_tokens=400):
        FakeJudge.calls.append((self.cache_dir, max_tokens))
        a = FakeJudge.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def make(tmp_path):
    FakeJudge.calls = []
    return T2.make_floor_judge(FakeJudge)(reasoning_effort="low", verbose=False, cache_dir=str(tmp_path / "jc"))


def test_retry_goes_through_its_own_cache_and_is_counted(tmp_path):
    J = make(tmp_path)
    FakeJudge.answers = ['{"revealed": [0', '{"revealed": [0], "satisfied": []}']
    T2.episode_begin()
    out = J.chat("s", "u", max_tokens=200)
    c = T2.episode_end()
    assert out.startswith('{"revealed": [0]')
    assert FakeJudge.calls[0] == (str(tmp_path / "jc"), T2.JUDGE_MIN_TOKENS)          # floor applied
    assert FakeJudge.calls[1] == (os.path.join(str(tmp_path / "jc"), "retry_%d" % T2.JUDGE_RETRY_TOKENS),
                                  T2.JUDGE_RETRY_TOKENS)                              # own cache, larger budget
    assert c["judge_retries"] == 1 and c["judge_unparseable"] == 0 and T2.episode_clean(c)


def test_error_and_empty_make_the_episode_unclean(tmp_path):
    J = make(tmp_path)
    FakeJudge.answers = [RuntimeError("timeout")]
    T2.episode_begin()
    with pytest.raises(RuntimeError):
        J.chat("s", "u")
    c = T2.episode_end()
    assert c["judge_error"] == 1 and not T2.episode_clean(c)
    FakeJudge.answers = ["", ""]
    T2.episode_begin()
    J.chat("s", "u")
    c = T2.episode_end()
    assert c["judge_empty"] == 1 and not T2.episode_clean(c)
