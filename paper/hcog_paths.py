"""Where the run grid is.

One resolver for every script that reads runs, rather than a copy of the same
fallback chain in each. Import it from anywhere under analysis/:

    import hcog_paths
    GRID = hcog_paths.grid_root()

Order of preference:

  $HCOG_GRID                      explicit override
  <repo>/results/<run>/phi4       the committed extract, ~10 MB, so a fresh
                                  clone reproduces every figure and table
  the raw grid on the data disk   2.6 GB, additionally holding the LLM prose
                                  and the per-step graphs; needed only by the
                                  integrity audit
"""
import os

RAW_DEFAULT = "/mnt/NewSSD/CS_project/neil/hcog_experiments"
_HERE = os.path.dirname(os.path.abspath(__file__))


def repo_root():
    """Hybrid-Network, from this file's location."""
    return os.path.normpath(os.path.join(_HERE, os.pardir))


def grid_root(run="M-1_main-grid", model="phi4"):
    env = os.environ.get("HCOG_GRID")
    if env:
        return env
    local = os.path.join(repo_root(), "results", run, model)
    if os.path.isdir(local):
        return local
    return os.path.join(RAW_DEFAULT, run, model)


def raw_grid_root(run="M-1_main-grid", model="phi4"):
    """The unabridged grid. Only the integrity audit needs this: it checks
    agents_data.json and edges_per_step.json, which the extract omits."""
    env = os.environ.get("HCOG_RAW_GRID")
    if env:
        return env
    return os.path.join(RAW_DEFAULT, run, model)


def bundle_file(name, run="M-1_main-grid"):
    """A per-grid derived table, e.g. neighbor_gap.csv."""
    return os.path.join(repo_root(), "results", run, name)
