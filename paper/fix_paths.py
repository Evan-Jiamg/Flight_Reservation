#!/usr/bin/env python3
"""fix_paths.py -- repoint every path that restructure.sh invalidated.

Run right after restructure.sh --apply. Every substitution is anchored on a
unique literal and fails loudly if it is not found, so a partial rename cannot
pass silently.

  python3 analysis/fix_paths.py          # show the diff
  python3 analysis/fix_paths.py --apply
"""
import argparse
import ast
import io
import os
import sys

ROOT = "/home/neil/Information_Management_Project/Echo-Chamber-Simulation/Hybrid-Network"

# file -> [(old, new), ...]
EDITS = {
    # ── data/ split into networks/ agents/ lexicons/ ──────────────────────
    "simulation/model.py": [
        ("os.path.dirname(__file__), '..', 'data', 'belief_keywords.json')",
         "os.path.dirname(__file__), '..', 'data', 'lexicons',\n"
         "                'belief_keywords.json')"),
        ("            os.path.dirname(__file__), '..', 'data',\n"
         "            f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}_{topic}.json'",
         "            os.path.dirname(__file__), '..', 'data', 'agents',\n"
         "            f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}_{topic}.json'"),
        ("            os.path.dirname(__file__), '..', 'data',\n"
         "            f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json'",
         "            os.path.dirname(__file__), '..', 'data', 'agents',\n"
         "            f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json'"),
        ("            os.path.dirname(__file__), '..', 'data',\n"
         "            f'{network_type}_network_num_agents_{num_agents}_seed_{seed}.json'",
         "            os.path.dirname(__file__), '..', 'data', 'networks',\n"
         "            f'{network_type}_network_num_agents_{num_agents}_seed_{seed}.json'"),
        ("                os.path.dirname(__file__), '..', 'data',\n"
         "                f'agents_backgrounds_num_agents_{num_agents}_{topic}_{backgrounds_label}.json'",
         "                os.path.dirname(__file__), '..', 'data', 'agents',\n"
         "                f'agents_backgrounds_num_agents_{num_agents}_{topic}_{backgrounds_label}.json'"),
        ("opinions_file = os.path.join(os.path.dirname(__file__), '..', 'opinions.json')",
         "opinions_file = os.path.join(os.path.dirname(__file__), '..', 'data',\n"
         "                                         'lexicons', 'opinions.json')"),
    ],
    "simulation/run_hybrid.py": [
        ("os.path.dirname(__file__), '..', 'data', 'belief_keywords.json')",
         "os.path.dirname(__file__), '..', 'data', 'lexicons',\n"
         "        'belief_keywords.json')"),
    ],
    "analysis/decompose_poa.py": [
        ('p = (PROJ_ROOT / "data" /',
         'p = (PROJ_ROOT / "data" / "agents" /'),
    ],
    "analysis/make_official_figs.py": [
        ('DATA = os.path.join(PROJ, "Hybrid-Network", "data")',
         'DATA = os.path.join(PROJ, "Hybrid-Network", "data", "agents")'),
    ],
    "plots/config.py": [
        ('DATA_DIR     = PROJ_ROOT / "data"',
         'DATA_DIR     = PROJ_ROOT / "data"\n'
         'NETWORK_DIR  = DATA_DIR / "networks"\n'
         'AGENT_DIR    = DATA_DIR / "agents"\n'
         'LEXICON_DIR  = DATA_DIR / "lexicons"'),
        ('    return ANALYSIS_DIR / f"timeseries_per_alpha_{topic}.json"',
         '    return ANALYSIS_DIR / "summaries" / f"timeseries_per_alpha_{topic}.json"'),
        ('    return ANALYSIS_DIR / f"summary_final_step_{topic}.json"',
         '    return ANALYSIS_DIR / "summaries" / f"summary_final_step_{topic}.json"'),
    ],
    # ── logs regrouped ────────────────────────────────────────────────────
    "scripts/monitor.sh": [
        ("Hybrid-Network/logs/run_vllm_*.log",
         "Hybrid-Network/logs/vllm/run_vllm_*.log"),
    ],
    # ── make_figures.py went one level down into analysis/legacy/ ─────────
    # (its GRID_ROOT is absolute, so only the grid pointer needs the change
    #  that every other analysis script gets below)
}

# The grid pointer. The committed bundle is preferred so a fresh clone works;
# the raw grid on the other disk is the fallback for anyone who has it.
GRID_OLD = '"/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"'
GRID_NEW = '''_grid_default()'''

GRID_HELPER = '''

def _grid_default():
    """Where the run grid is.

    Prefer the committed bundle: it is ~10 MB of exactly what the analysis
    reads, so a fresh clone works without the 2.6 GB raw tree. Fall back to the
    raw grid on the data disk, which additionally holds the LLM prose that no
    figure consumes. Override with $HCOG_GRID.
    """
    env = os.environ.get("HCOG_GRID")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for up in ("..", "../.."):                 # analysis/ and analysis/legacy/
        local = os.path.normpath(
            os.path.join(here, up, "results", "M-1_main-grid", "phi4"))
        if os.path.isdir(local):
            return local
    return "/mnt/NewSSD/CS_project/neil/hcog_experiments/M-1_main-grid/phi4"

'''

GRID_FILES = ["analysis/make_official_figs.py",
              "analysis/make_convergence_figs.py",
              "analysis/export_converged_table.py"]


def apply_edits(path, edits, dry):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return "missing"
    s = io.open(full, encoding="utf-8").read()
    n = 0
    for old, new in edits:
        if old not in s:
            return "NOT FOUND: " + old.strip().splitlines()[0][:70]
        s = s.replace(old, new, 1)
        n += 1
    if path.endswith(".py"):
        ast.parse(s)
    if not dry:
        io.open(full, "w", encoding="utf-8", newline="\n").write(s)
    return "%d edit(s)" % n


def apply_grid(path, dry):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return "missing"
    s = io.open(full, encoding="utf-8").read()
    if GRID_OLD not in s:
        return "NOT FOUND: grid literal"
    s = s.replace(GRID_OLD, GRID_NEW, 1)
    # drop the helper in just above the first use
    marker = "GRID = "
    i = s.index(marker)
    line_start = s.rindex("\n", 0, i) + 1
    s = s[:line_start] + GRID_HELPER.lstrip("\n") + "\n" + s[line_start:]
    ast.parse(s)
    if not dry:
        io.open(full, "w", encoding="utf-8", newline="\n").write(s)
    return "grid pointer + helper"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry = not args.apply

    bad = 0
    print("== path edits ==")
    for path, edits in EDITS.items():
        r = apply_edits(path, edits, dry)
        flag = "" if r[0].isdigit() else "  <-- PROBLEM"
        if flag:
            bad += 1
        print("  %-38s %s%s" % (path, r, flag))

    print()
    print("== grid pointer ==")
    for path in GRID_FILES:
        r = apply_grid(path, dry)
        flag = "" if r.startswith("grid") else "  <-- PROBLEM"
        if flag:
            bad += 1
        print("  %-38s %s%s" % (path, r, flag))

    print()
    if bad:
        print("%d file(s) failed; nothing written" % bad)
        return 1
    print("dry run -- pass --apply" if dry else "applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
