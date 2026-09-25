import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import batching


def test_results_go_back_to_their_callers_and_batches_form():
    lock = threading.Lock()
    seen_batches = []

    def fn(items):
        assert lock.locked()                          # runs under the GPU lock
        seen_batches.append(list(items))
        time.sleep(0.02)
        return [x * 10 for x in items]

    b = batching.Batcher(fn, lock, max_batch=8, window_s=0.05)
    with ThreadPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(b, range(16)))
    assert out == [x * 10 for x in range(16)]
    assert max(len(x) for x in seen_batches) > 1 and all(len(x) <= 8 for x in seen_batches)
    assert b.stats()["items"] == 16
    b.close()


def test_grouping_by_key_never_mixes_settings():
    lock = threading.Lock()

    def fn(items):
        assert len({it["t"] for it in items}) == 1, "mixed keys in one call"
        return [it["v"] for it in items]

    b = batching.Batcher(fn, lock, key=lambda it: it["t"], max_batch=8, window_s=0.05)
    items = [{"t": i % 3, "v": i} for i in range(24)]
    with ThreadPoolExecutor(max_workers=24) as ex:
        out = list(ex.map(b, items))
    assert out == list(range(24))
    b.close()


def test_map_keeps_order_and_exceptions_reach_every_caller():
    lock = threading.Lock()
    b = batching.Batcher(lambda items: [x + 1 for x in items], lock, max_batch=4, window_s=0.01)
    assert b.map([5, 1, 9, 3, 7]) == [6, 2, 10, 4, 8]
    b.close()

    def boom(items):
        raise RuntimeError("cuda oom")
    b2 = batching.Batcher(boom, lock, max_batch=4, window_s=0.01)
    fs = [b2.submit(i) for i in range(3)]
    for f in fs:
        with pytest.raises(RuntimeError):
            f.result(timeout=5)
    b2.close()


def test_wrong_result_count_is_an_error():
    lock = threading.Lock()
    b = batching.Batcher(lambda items: items[:-1], lock, max_batch=4, window_s=0.05)
    fs = [b.submit(i) for i in range(3)]
    with pytest.raises(AssertionError):
        [f.result(timeout=5) for f in fs]
    b.close()
