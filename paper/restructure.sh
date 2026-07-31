#!/usr/bin/env bash
# restructure.sh -- lay Hybrid-Network out for a paper repository.
#
# Grouping only. No source file is rewritten here; the path references that the
# moves invalidate are fixed separately by fix_paths.py, which runs after.
#
# The top-level code packages (core/ simulation/ plots/ reddit/ analysis/) stay
# where they are on purpose. Every module inside them reaches its data through
# os.path.dirname(__file__)/'..', so burying them one level deeper under a src/
# would silently redirect roughly two dozen paths in model.py alone. Grouping
# that breaks the thing being grouped is not tidier.
#
#   ./restructure.sh          # print the plan
#   ./restructure.sh --apply
set -euo pipefail

ROOT="/home/neil/Information_Management_Project/Echo-Chamber-Simulation/Hybrid-Network"
APPLY="${1:-}"
cd "$ROOT"

say() { printf '  %s\n' "$*"; }
run() { if [ "$APPLY" = "--apply" ]; then eval "$@"; fi; }

echo "== delete: dead scripts =="
# both point at /home/evan-jiamg/... and dynamic_network/hybrid/run_hybrid.py,
# neither of which exists; superseded by simulation/run_all_parallel.py
say "run_sweep_10seeds.sh          (dead host + layout)"
say "run_sweep_25steps.sh          (dead host + layout)"
# calls main.py, which is not in this tree, on topic 'euthanasia', not in this paper
say "scripts/run_experiments.sh    (dead entry point)"
run "rm -f run_sweep_10seeds.sh run_sweep_25steps.sh scripts/run_experiments.sh"

echo
echo "== delete: build artefacts =="
say "$(find . -name __pycache__ -type d -not -path './ops/*' | wc -l) __pycache__ directories"
run "find . -name __pycache__ -type d -not -path './ops/*' -exec rm -rf {} + 2>/dev/null || true"

echo
echo "== data/: 96 network files off the top level =="
say "data/*_network_num_agents_50_seed_*.json -> data/networks/"
run "mkdir -p data/networks"
run "mv data/*_network_num_agents_50_seed_*.json data/networks/ 2>/dev/null || true"

# 10 of the 13 differ from the current files: they are the scale_free networks
# as they stood before the 07-25 regeneration, so the June runs were made with
# them. The 3 identical ones carry nothing.
say "data/_networks_backup_pre_rerun/ -> data/networks/_superseded_pre_rerun/ (drop 3 duplicates)"
run "mkdir -p data/networks/_superseded_pre_rerun"
run "mv data/_networks_backup_pre_rerun/*.json data/networks/_superseded_pre_rerun/ 2>/dev/null || true"
run "rmdir data/_networks_backup_pre_rerun 2>/dev/null || true"
for n in random scale_free small_world; do
    f="data/networks/_superseded_pre_rerun/${n}_network_num_agents_50_seed_50.json"
    c="data/networks/${n}_network_num_agents_50_seed_50.json"
    if [ -f "$f" ] && [ -f "$c" ] && cmp -s "$f" "$c"; then
        run "rm -f '$f'"
    fi
done

say "agent inputs -> data/agents/"
run "mkdir -p data/agents"
run "mv data/agents_backgrounds_*.json data/agents/ 2>/dev/null || true"
run "mv data/numeric_sim_opnions_*.json data/agents/ 2>/dev/null || true"

say "prompt/scoring vocabularies -> data/lexicons/  (incl. opinions.json from the root)"
run "mkdir -p data/lexicons"
run "mv data/belief_keywords.json data/mitigation_perspectives.json data/lexicons/ 2>/dev/null || true"
run "mv opinions.json data/lexicons/ 2>/dev/null || true"

echo
echo "== analysis/: separate scripts, aggregates and tools =="
say "aggregated JSON -> analysis/summaries/"
run "mkdir -p analysis/summaries"
run "mv analysis/timeseries_per_alpha_*.json analysis/summary_final_step_*.json analysis/summaries/ 2>/dev/null || true"
run "mv analysis/summary_partial.json analysis/timeseries_partial.json analysis/summaries/ 2>/dev/null || true"
run "mv analysis/poa_decomposition_M-1_main-grid.json analysis/summaries/ 2>/dev/null || true"
run "mv roberta_vs_llm_comparison.json analysis/summaries/ 2>/dev/null || true"

say "figure plumbing -> analysis/tools/"
run "mkdir -p analysis/tools"
run "mv analysis/validate_eps.py analysis/eps_from_pdf.sh analysis/crop_pdf.sh analysis/tools/ 2>/dev/null || true"
run "mv analysis/cleanup_figures.sh analysis/tools/ 2>/dev/null || true"

# same five figures as make_official_figs.py, plus three the paper dropped
say "make_figures.py -> analysis/legacy/  (superseded by make_official_figs.py)"
run "mkdir -p analysis/legacy"
run "mv analysis/make_figures.py analysis/legacy/ 2>/dev/null || true"

echo
echo "== root: sweeps and logs off the top level =="
say "run_sweep_{abortion,gun_control}.sh -> scripts/"
run "mv run_sweep_abortion.sh run_sweep_gun_control.sh scripts/ 2>/dev/null || true"

say "logs -> logs/vllm/ and logs/sweeps/"
run "mkdir -p logs/vllm logs/sweeps"
run "mv logs/run_vllm_*.log logs/vllm/ 2>/dev/null || true"
run "mv sweep_abortion.log sweep_gun_control.log sweep_log.txt logs/sweeps/ 2>/dev/null || true"
run "mv init_abortion.log init_gun_control.log logs/sweeps/ 2>/dev/null || true"

echo
if [ "$APPLY" != "--apply" ]; then
    echo "dry run -- pass --apply"
else
    echo "moved. now run: python3 analysis/fix_paths.py --apply"
fi
