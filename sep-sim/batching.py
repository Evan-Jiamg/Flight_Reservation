# -*- coding: utf-8 -*-
"""Cross-thread dynamic batching for GPU generation (pure python; no torch at import).

Episodes run in threads (train_planner_rl --rollout-workers). Without batching every Planner / Ditto
generation is serialised behind one GPU lock, so N concurrent episodes cost N sequential generations.
A Batcher collects the requests that arrive within `window_s` (up to `max_batch`), groups them by
`key(item)` (requests in one generate call must share sampling settings) and runs `fn(items)` under
the GPU lock; each caller gets its own result back through a Future.

Semantics are the caller's `fn`; the batcher only changes WHEN things run. It never reorders a
caller's own requests relative to each other (a caller blocks on its future).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future


class Batcher:
    def __init__(self, fn, lock, key=lambda item: None, max_batch=8, window_s=0.05, name="batcher"):
        self.fn, self.lock, self.key = fn, lock, key
        self.max_batch, self.window_s = int(max_batch), float(window_s)
        self._q = []
        self._cv = threading.Condition()
        self._stop = False
        self.n_calls = self.n_items = 0
        self.max_seen = 0
        self._t = threading.Thread(target=self._loop, name=name, daemon=True)
        self._t.start()

    def submit(self, item):
        f = Future()
        with self._cv:
            self._q.append((item, f))
            self._cv.notify()
        return f

    def __call__(self, item):
        return self.submit(item).result()

    def map(self, items):
        fs = [self.submit(x) for x in items]
        return [f.result() for f in fs]

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(timeout=5)

    def _take(self):
        with self._cv:
            while not self._q and not self._stop:
                self._cv.wait()
            if self._stop and not self._q:
                return None
        # let concurrent callers join the batch
        deadline = time.time() + self.window_s
        while time.time() < deadline:
            with self._cv:
                if len(self._q) >= self.max_batch:
                    break
            time.sleep(0.005)
        with self._cv:
            k0 = self.key(self._q[0][0])
            batch, rest = [], []
            for it in self._q:
                (batch if (self.key(it[0]) == k0 and len(batch) < self.max_batch) else rest).append(it)
            self._q = rest
        return batch

    def _loop(self):
        while True:
            batch = self._take()
            if batch is None:
                return
            items = [it for it, _ in batch]
            try:
                with self.lock:
                    outs = self.fn(items)
                if len(outs) != len(items):
                    raise AssertionError("batched fn returned %d results for %d items" % (len(outs), len(items)))
                for (_, f), o in zip(batch, outs):
                    f.set_result(o)
            except BaseException as e:           # every waiting caller sees the failure
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
            self.n_calls += 1
            self.n_items += len(items)
            self.max_seen = max(self.max_seen, len(items))

    def stats(self):
        return {"calls": self.n_calls, "items": self.n_items, "max_batch_seen": self.max_seen,
                "mean_batch": (self.n_items / self.n_calls) if self.n_calls else 0.0}
