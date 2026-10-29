#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — full-set evaluation of one agent, on a cluster or on a laptop
#
# DUAL-MODE, one script:
#   sbatch hpc/run_eval.sh                  -> one node, every core filled
#   sbatch --array=0-15 hpc/run_eval.sh     -> spread across array tasks instead
#   bash   hpc/run_eval.sh                  -> local run, WORKERS parallel processes
#
# The same file works in all three so the thing that was tested locally is the thing
# that runs on the cluster, rather than a second script that drifts from the first.
#
# WHY ONE NODE IS THE DEFAULT. The work is ~11 core-hours of independent scenarios. An
# array of single-core tasks finishes just as fast, but if the site hands out whole
# nodes, each task is billed as an entire node however few cores it uses, and the same
# work is charged many times over. Filling one node is charged once and needs no
# knowledge of local accounting rules.
#
# RESUMABLE. A worker whose output file already exists is skipped, so a job that hit a
# time limit, a failed node, or a requeue can simply be resubmitted and it continues.
# To force recomputation, delete the corresponding results/*.json.
#
# Environment variables:
#   SUBMISSION  agent folder under agent/            (default xdp_agent)
#   ENVNAME     grid2op environment                  (default l2rpn_idf_2023_train)
#   TOTAL       number of scenarios                  (default 764)
#   WORKERS     parallel processes when run locally  (default 4)
#   OUTDIR      results directory                    (default results)
#   SEEDS       JSON list of per-scenario env seeds   (optional; re-draws opponent attacks)
#   EPSFILE     JSON list of episode ids to run       (optional; a SUBSET of the env, e.g. only the
#               chronics of a dataset that appear in no other dataset). When given, TOTAL is ignored
#               and the ids are sliced across workers instead of a contiguous range.
#
# CONCURRENT ARMS. Several arms may be submitted at once; they are independent jobs. Each must be given
# its own OUTDIR, which separates both the result files and -- because the per-worker log name is derived
# from it -- the logs. Without that derivation every arm would write logs/worker_000.log and a failed
# slice could not be traced back to the arm that produced it.
# =============================================================================
#SBATCH --job-name=l2rpn-eval
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
# Add --mail-type=END,FAIL --mail-user=... on the sbatch command line if you want
# notifications; it is left off here so no address is written into a shared file.

set -euo pipefail

ts()      { date '+%H:%M:%S'; }
elapsed() { printf '%dh%02dm%02ds' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

# --- environment: only touch modules when actually running under the scheduler -------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Site-specific setup: load your cluster's Python module and activate the environment built from
    # requirements.txt, e.g. `module load Python/3.11` and `source "$HOME/venvs/xdp/bin/activate"`.
    if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
    export GRID2OP_DATA_PATH="${GRID2OP_DATA_PATH:-$HOME/data_grid2op}"
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
else
    cd "$(cd "$(dirname "$0")/.." && pwd)"      # the repository root, from this script's location
fi

export OMP_NUM_THREADS=1          # one core per worker; the power-flow backend must not oversubscribe
export MPLBACKEND=Agg             # never try to open a display on a headless node

SUB="${SUBMISSION:-xdp_agent}"
ENVNAME="${ENVNAME:-l2rpn_idf_2023_train}"
TOTAL="${TOTAL:-764}"
OUTDIR="${OUTDIR:-results}"
SEEDS="${SEEDS:-}"
EPSFILE="${EPSFILE:-}"
mkdir -p "$OUTDIR" logs

# --- how many workers, and which slice each takes ------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    NWORK="${SLURM_ARRAY_TASK_COUNT:-1}"; ONLY="${SLURM_ARRAY_TASK_ID}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    # SLURM's own count is authoritative. Do NOT use bare `nproc` here: it honours OMP_NUM_THREADS,
    # which is set to 1 just above so each worker stays single-threaded, so `nproc` reports 1 and the
    # whole run collapses onto a single worker that silently takes days instead of minutes.
    NWORK="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"; ONLY=""
else
    NWORK="${WORKERS:-$(nproc --all)}"; ONLY=""
fi
# An explicit id list replaces the contiguous range: the pools this battery uses are SUBSETS of an
# environment (the chronics of one dataset that appear in no other), which a [from,to) slice cannot express.
if [ -n "$EPSFILE" ]; then
    TOTAL=$(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$EPSFILE")
fi
PER=$(( (TOTAL + NWORK - 1) / NWORK ))
echo "[$(ts)] agent=$SUB env=$ENVNAME scenarios=$TOTAL workers=$NWORK (~$PER each)${EPSFILE:+ subset=$EPSFILE}"

run_worker() {
    local i=$1
    local from=$(( i * PER )) to=$(( i * PER + PER ))
    [ "$to" -gt "$TOTAL" ] && to=$TOTAL
    [ "$from" -ge "$TOTAL" ] && return 0
    local out; out="$OUTDIR/${SUB}_w$(printf '%03d' "$i").json"
    # Resume guard: a slice that already produced its file is left alone.
    if [ -s "$out" ]; then
        echo "[$(ts)] SKIP  worker $i  [$from,$to)  — $out already present"
        return 0
    fi
    local t0; t0=$(date +%s)
    echo "[$(ts)] START worker $i  scenarios [$from,$to)"
    # SEEDS (optional): a JSON list of env seeds, one per scenario index. Supplying it re-draws the
    # OPPONENT's attacks on identical load data, which is the only axis on which a fresh sample can be
    # taken from chronics we already hold -- the load series are files, the adversary is not.
    if [ -n "$EPSFILE" ]; then
        # This worker's share of the id list, written to its own file so eval_full sees a plain list.
        local slice="$OUTDIR/.eps_w$(printf '%03d' "$i").json"
        python -c "import json,sys;ids=json.load(open(sys.argv[1]));json.dump(ids[int(sys.argv[2]):int(sys.argv[3])],open(sys.argv[4],'w'))" \
            "$EPSFILE" "$from" "$to" "$slice"
        python evaluation/eval_full.py --submission "$SUB" --env "$ENVNAME" \
            --eps "$slice" --mode fast --reference --trace \
            ${SEEDS:+--seeds "$SEEDS"} \
            --out "$out" > "logs/$(basename "$OUTDIR")_worker_$(printf '%03d' "$i").log" 2>&1
    else
        python evaluation/eval_full.py --submission "$SUB" --env "$ENVNAME" \
            --from "$from" --to "$to" --mode fast --reference --trace \
            ${SEEDS:+--seeds "$SEEDS"} \
            --out "$out" > "logs/$(basename "$OUTDIR")_worker_$(printf '%03d' "$i").log" 2>&1
    fi
    echo "[$(ts)] DONE  worker $i  $(elapsed $(( $(date +%s) - t0 )))"
}
export -f run_worker elapsed ts
export SUB ENVNAME TOTAL OUTDIR PER SEEDS EPSFILE

if [ -n "$ONLY" ]; then
    run_worker "$ONLY"                                   # one array task = one slice
else
    seq 0 $((NWORK - 1)) | xargs -P "$NWORK" -I{} bash -c 'run_worker {}'
fi

echo "[$(ts)] finished — $(ls "$OUTDIR"/*.json 2>/dev/null | wc -l) result files in $OUTDIR"
