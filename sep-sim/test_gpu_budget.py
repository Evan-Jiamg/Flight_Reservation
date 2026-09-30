"""User 2026-09-30 ("不能被搶卡，先佔資源"): the training process holds its whole GPU budget from build on
(rl_algos.reserve_gpu_budget, Trainer.keep_budget, --gpu-budget-gib). Tested against a mock of torch's caching allocator
(no GPU): freed blocks stay reserved unless the cache is emptied."""
import json
import re
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_algos as RA  # noqa: E402
import train_planner_rl as T  # noqa: E402

GIB = RA.GIB


class _Cuda:
    """A caching allocator: `reserved` (held from the driver) >= `allocated` (live tensors); an allocation is served from
    the cached part first, else takes new memory from `free` (the GPU's free memory), else raises "CUDA out of memory".
    caching=False models an allocator that returns freed memory at once (what empty_cache would do)."""

    def __init__(self, reserved, allocated, free, caching=True, fail_after=None):
        self.reserved, self.allocated, self.free, self.caching = reserved, allocated, free, caching
        self.fail_after, self.n_alloc = fail_after, 0

    def memory_reserved(self, gpu):
        return self.reserved

    def memory_allocated(self, gpu):
        return self.allocated

    def mem_get_info(self, gpu):
        return self.free, 80 * GIB

    def alloc(self, n):
        self.n_alloc += 1
        if self.fail_after is not None and self.n_alloc > self.fail_after:
            raise RuntimeError("CUDA out of memory. Tried to allocate %d bytes" % n)
        cached = self.reserved - self.allocated
        if n > cached:
            need = n - cached
            if need > self.free:
                raise RuntimeError("CUDA out of memory. Tried to allocate %d bytes" % n)
            self.reserved += need
            self.free -= need
        self.allocated += n

    def release(self, n):
        self.allocated -= n
        if not self.caching:
            self.free += n
            self.reserved -= n


class _Tensor:
    def __init__(self, cuda, n):
        self.cuda, self.n = cuda, n
        cuda.alloc(n)

    def __del__(self):
        self.cuda.release(self.n)


def mock_torch(**kw):
    cuda = _Cuda(**kw)
    t = types.SimpleNamespace(cuda=cuda, uint8="uint8",
                              empty=lambda n, dtype=None, device=None: _Tensor(cuda, int(n)))
    return t, cuda


def test_reserves_up_to_budget_minus_margin_and_keeps_it():
    torch, cuda = mock_torch(reserved=37 * GIB, allocated=36 * GIB, free=30 * GIB)
    logs = []
    info = RA.reserve_gpu_budget(1, 43, margin_gib=0.5, torch_mod=torch, log=logs.append)
    assert cuda.reserved >= int(42.5 * GIB) and cuda.reserved <= 43 * GIB      # held, never beyond the budget
    assert cuda.allocated == 36 * GIB                                          # the reserve tensors were freed ...
    assert cuda.reserved - cuda.allocated >= int(5.5 * GIB) - 1                # ... into the cache: still ours
    assert info["reserved_after"] == cuda.reserved and info["small_blocks"] == 1024 and not info["over_budget"]
    assert "reserved 42.50 GiB" in logs[-1]
    # a second call is a no-op (already at the target)
    n = cuda.n_alloc
    RA.reserve_gpu_budget(1, 43, torch_mod=torch, log=logs.append)
    assert cuda.n_alloc == n


class _FragCuda:
    """A caching allocator with NON-fungible free fragments (audit S1): an allocation is served from ONE cached fragment
    at least as large (the rest of that fragment stays cached), else maps new memory; a freed tensor becomes its own
    fragment (no merging)."""

    def __init__(self, reserved, allocated, fragments, free):
        assert sum(fragments) == reserved - allocated
        self.reserved, self.allocated, self.frags, self.free = reserved, allocated, list(fragments), free
        self.n_alloc = 0

    def memory_reserved(self, gpu):
        return self.reserved

    def memory_allocated(self, gpu):
        return self.allocated

    def mem_get_info(self, gpu):
        return self.free, 80 * GIB

    def alloc(self, n):
        self.n_alloc += 1
        fit = [i for i, f in enumerate(self.frags) if f >= n]
        if fit:
            i = min(fit, key=lambda j: self.frags[j])
            self.frags[i] -= n
        else:
            if n > self.free:
                raise RuntimeError("CUDA out of memory. Tried to allocate %d bytes" % n)
            self.free -= n
            self.reserved += n
        self.allocated += n

    def release(self, n):
        self.allocated -= n
        self.frags.append(n)


def test_scattered_fragments_never_push_the_reservation_past_the_budget():
    """30 GiB reserved, 22 GiB live, 8 GiB cached in 64 MiB fragments (none fits a 256 MiB chunk): a single
    "target - allocated" tensor would map ~20 GiB more (reserved ~50 GiB); the chunked reserve stops at the budget."""
    cuda = _FragCuda(reserved=30 * GIB, allocated=22 * GIB, fragments=[64 * 1024 * 1024] * 128, free=40 * GIB)
    torch = types.SimpleNamespace(cuda=cuda, uint8="uint8", empty=lambda n, dtype=None, device=None: _Tensor(cuda, int(n)))
    logs = []
    info = RA.reserve_gpu_budget(0, 43, margin_gib=0.5, torch_mod=torch, log=logs.append)
    assert int(42.5 * GIB) <= cuda.reserved <= 43 * GIB, cuda.reserved / GIB
    assert not info["over_budget"] and cuda.allocated == 22 * GIB and not any("WARNING" in l for l in logs)
    # one fragment large enough: it is used first, nothing new is mapped for it
    cuda2 = _FragCuda(reserved=42 * GIB, allocated=22 * GIB, fragments=[20 * GIB], free=40 * GIB)
    torch2 = types.SimpleNamespace(cuda=cuda2, uint8="uint8", empty=lambda n, dtype=None, device=None: _Tensor(cuda2, int(n)))
    RA.reserve_gpu_budget(0, 43, margin_gib=0.5, torch_mod=torch2, log=logs.append)
    assert int(42.5 * GIB) <= cuda2.reserved <= 43 * GIB


def test_already_above_budget_is_logged_not_shrunk():
    torch, cuda = mock_torch(reserved=44 * GIB, allocated=44 * GIB, free=0)
    logs = []
    info = RA.reserve_gpu_budget(0, 43, torch_mod=torch, log=logs.append)
    assert info["over_budget"] and info["added"] == 0 and "WARNING" in logs[-1] and cuda.n_alloc == 0


def test_budget_not_free_fails_loudly():
    torch, cuda = mock_torch(reserved=37 * GIB, allocated=36 * GIB, free=2 * GIB)     # someone took the rest
    logs = []
    with pytest.raises(RA.GpuBudgetError) as e:
        RA.reserve_gpu_budget(0, 43, torch_mod=torch, log=logs.append)
    assert "GPU budget not available" in str(e.value) and "CUDA out of memory" in str(e.value)
    assert cuda.n_alloc == 0 and logs and "GPU budget not available" in logs[-1]


def test_allocation_failure_while_reserving_fails_loudly():
    torch, cuda = mock_torch(reserved=37 * GIB, allocated=36 * GIB, free=30 * GIB, fail_after=100)
    with pytest.raises(RA.GpuBudgetError) as e:
        RA.reserve_gpu_budget(0, 43, torch_mod=torch, log=lambda s: None)
    assert "allocation failed" in str(e.value) and "CUDA out of memory" in str(e.value)
    assert cuda.allocated == 36 * GIB                                          # nothing left allocated


def test_a_reservation_that_does_not_hold_fails():
    torch, cuda = mock_torch(reserved=37 * GIB, allocated=36 * GIB, free=30 * GIB, caching=False)
    with pytest.raises(RA.GpuBudgetError) as e:
        RA.reserve_gpu_budget(0, 43, torch_mod=torch, log=lambda s: None)
    assert "did not hold" in str(e.value)


def test_keep_budget_logs_and_stops_with_exit_75(tmp_path, monkeypatch):
    tr = T.Trainer.__new__(T.Trainer)
    tr.a = types.SimpleNamespace(dry_run=False, gpu=2, gpu_budget_gib=43.0, out=str(tmp_path))
    monkeypatch.setattr(RA, "reserve_gpu_budget", lambda gpu, b: {"gpu": gpu, "budget_gib": b, "reserved_after": 1})
    assert tr.keep_budget("build")["gpu"] == 2

    def gone(gpu, b):
        raise RA.GpuBudgetError("GPU budget not available on GPU 2: ... CUDA out of memory while reserving")
    monkeypatch.setattr(RA, "reserve_gpu_budget", gone)
    with pytest.raises(SystemExit) as e:
        tr.keep_budget("update u3")
    assert e.value.code == RA.GPU_BUDGET_EXIT == 75
    rows = [json.loads(l) for l in open(tmp_path / "gpu_budget.jsonl")]
    assert [r["where"] for r in rows] == ["build", "update u3"] and "GPU budget not available" in rows[1]["error"]
    tr.a = types.SimpleNamespace(dry_run=True, gpu=0, gpu_budget_gib=43.0, out=str(tmp_path))
    assert tr.keep_budget("x") is None                                          # dry run: nothing to reserve
    tr.a = types.SimpleNamespace(dry_run=False, gpu=0, gpu_budget_gib=0.0, out=str(tmp_path))
    assert tr.keep_budget("x") is None                                          # --gpu-budget-gib 0: off


def test_flag_gate_and_resume():
    base = ["--fold", "2", "--out", "o", "--planner-path", "Qwen3-4B-Instruct-2507"]
    assert T.parse_args(base).gpu_budget_gib == 43.0
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--gpu-budget-gib", "40"])                          # another budget on a real run: ablation
    assert T.parse_args(base + ["--gpu-budget-gib", "40", "--ablation", "small-gpu"]).gpu_budget_gib == 40.0
    with pytest.raises(SystemExit):
        T.parse_args(base + ["--gpu-budget-gib", "81", "--ablation", "x"])
    assert T.parse_args(["--dry-run", "--fold", "0", "--out", "o", "--gpu-budget-gib", "10"]).gpu_budget_gib == 10.0
    assert "gpu_budget_gib" in T.RESUME_MAY_CHANGE                              # a resource setting: may change on resume


def test_no_empty_cache_on_the_run_path():
    """Nothing on the v17 run path empties the allocator cache (it would release the reserved budget)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("train_planner_rl.py", "rl_algos.py", "task2_env.py", "ditto_e16.py", "eval_test_rl.py", "smoke_v17.py",
              "vllm_planner.py", "batching.py", "implicit_profile.py", "style_select.py", "fit_prompts.py"):
        src = open(os.path.join(here, f), encoding="utf-8").read()
        calls = [l for l in src.splitlines() if re.match(r"\s*(torch\.cuda\.)?empty_cache\(", l)]      # a call statement
        assert not calls, (f, calls)
