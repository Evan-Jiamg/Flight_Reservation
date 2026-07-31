#!/usr/bin/env bash
# cleanup_figures.sh -- drop every figure except the paper set.
#
# Figures are build output. The inputs they are drawn from are kept, and every
# command needed to rebuild them is in analysis/REGENERATING_FIGURES.md, which
# is written before this runs.
#
# analysis/figures/official_paper/ is never touched.
#
# A manifest of everything removed (path, size, mtime, sha256) is written to
# analysis/figures/DELETED_MANIFEST.txt so the removal stays auditable. The
# tracked files among them are also still in git history.
#
#   ./cleanup_figures.sh          # show what would go
#   ./cleanup_figures.sh --apply  # actually delete
set -euo pipefail

ROOT="/home/neil/Information_Management_Project/Echo-Chamber-Simulation/Hybrid-Network"
APPLY="${1:-}"
MANIFEST="$ROOT/analysis/figures/DELETED_MANIFEST.txt"

cd "$ROOT"

# Everything below is regenerable; the trailing comment says how.
mapfile -t TARGETS < <(
    # per-run metric charts, rewritten by simulation/model.py on the next run;
    # the filename has no seed, so each only ever held the last seed to finish
    find analysis/figures -maxdepth 1 -type f -name 'hybrid_*_agents50.png'
    # same three figures as official_paper/, written here by the default outdir
    find analysis/figures -maxdepth 1 -type f -name 'convergence_*'
    # plots/plot_convergence_multiples.py
    find analysis/figures/timeseries -type f 2>/dev/null
    # plots/plot_stance_distribution.py
    find analysis/figures/reddit_only -type f 2>/dev/null
    # stale 07-29 copies, predating the EPS and colour work
    find figures -type f 2>/dev/null
    # backup taken before cropping; the cropped PDF still carries the draw.io
    # source XML in its Subject metadata, so this adds nothing
    find analysis/figures/official_paper -maxdepth 1 -name 'Pipeline_uncropped.pdf'
)

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "nothing to remove"; exit 0
fi

total=$(du -ch "${TARGETS[@]}" 2>/dev/null | tail -1 | cut -f1)
echo "${#TARGETS[@]} files, $total"
printf '%s\n' "${TARGETS[@]}" | sed 's|^|  |' | head -12
[ "${#TARGETS[@]}" -gt 12 ] && echo "  ... and $(( ${#TARGETS[@]} - 12 )) more"

# never let official_paper figures into the list
if printf '%s\n' "${TARGETS[@]}" | grep -v 'Pipeline_uncropped' | grep -q 'official_paper'; then
    echo "REFUSING: an official_paper figure is in the delete list"; exit 1
fi

if [ "$APPLY" != "--apply" ]; then
    echo
    echo "dry run -- pass --apply to delete"
    exit 0
fi

{
    echo "# figures removed $(date -Iseconds)"
    echo "# rebuild instructions: analysis/REGENERATING_FIGURES.md"
    echo "# tracked files are also recoverable with: git checkout HEAD -- <path>"
    echo "#"
    printf '%-72s %10s  %-16s  %s\n' "path" "bytes" "mtime" "sha256"
    for f in "${TARGETS[@]}"; do
        printf '%-72s %10s  %-16s  %s\n' \
            "$f" "$(stat -c%s "$f")" "$(stat -c%y "$f" | cut -d. -f1 | tr ' ' 'T')" \
            "$(sha256sum "$f" | cut -c1-16)"
    done
} > "$MANIFEST"

rm -f "${TARGETS[@]}"
rmdir analysis/figures/timeseries analysis/figures/reddit_only figures 2>/dev/null || true

echo
echo "removed ${#TARGETS[@]} files; manifest at ${MANIFEST#$ROOT/}"
echo "kept: $(find analysis/figures/official_paper -type f | wc -l) files in analysis/figures/official_paper"
