#!/bin/bash
# Runs test_ep.py in perf mode with the two probes compiled in, on every node of the job, four times:
# two shapes (FP8 hidden 7168 top-6 8192 tokens, BF16 hidden 3328 top-6 4096 tokens), each quiet and
# under the EFA burst stressor bg_a2a.py started in the same step. Prints one summary line per run.
#
#   TREE=<this checkout> SQSH=<container image with torch, NCCL and the EFA plugin> \
#       sbatch -N <nodes> --job-name=<name> tests/probes/exit_flush/run-probe.sbatch
#   OUT      results directory (default $TREE/tests/probes/exit_flush/results/<utc>-<job>; mounted into the container)
#   BG_DUR   seconds the stressor runs (default 420), BG_LEAD seconds it runs before the test starts (default 45)
#   NUM_SMS  --num-sms of the test (default 28)
#   The deep_ep extension is built in $TREE when it is missing (TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3").
#SBATCH -t 01:00:00
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
set -u
TREE=${TREE:?path of the DeepEP checkout}
SQSH=${SQSH:?container image}
OUT=${OUT:-$TREE/tests/probes/exit_flush/results/$(date -u +%Y%m%dT%H%M%SZ)-$SLURM_JOB_ID}
BG_DUR=${BG_DUR:-420}; BG_LEAD=${BG_LEAD:-45}; NUM_SMS=${NUM_SMS:-28}
HERE=$TREE/tests/probes/exit_flush
mkdir -p "$OUT/runs"
exec > >(tee -a "$OUT/log.txt") 2>&1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
N=$SLURM_JOB_NUM_NODES
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$OUT/nodes.txt"
log "job=$SLURM_JOB_ID nodes=$N image=$SQSH tree=$TREE"
export MASTER_ADDR=$(head -n1 "$OUT/nodes.txt") WORLD_SIZE=$N
MOUNTS=/dev/infiniband:/dev/infiniband,/dev/gdrdrv:/dev/gdrdrv,$TREE:$TREE
case "$OUT" in "$TREE"/*) ;; *) MOUNTS=$MOUNTS,$OUT:$OUT ;; esac

srun -N "$N" --ntasks-per-node=1 bash -c 'echo "$(hostname) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " ")"' | tee "$OUT/preflight.txt"
if awk '{for(i=2;i<=NF;i++) if($i>500) bad=1} END{exit bad}' "$OUT/preflight.txt"; then log "preflight ok"; else log "preflight FAILED: GPU memory in use"; exit 2; fi

if ! ls "$TREE"/deep_ep/_C*.so >/dev/null 2>&1; then
    srun -N1 --ntasks=1 --container-image "$SQSH" --container-mounts "$MOUNTS" bash -lc '
        cd '"$TREE"' && rm -rf build dist *.egg-info
        export TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3" DISABLE_AGGRESSIVE_PTX_INSTRS=1
        export LIBRARY_PATH=$(python3 -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\"))")${LIBRARY_PATH:+:$LIBRARY_PATH}
        python3 setup.py build && cp build/lib.linux-x86_64-cpython-*/deep_ep/_C*.so deep_ep/ \
        && PYTHONPATH='"$TREE"' python3 -c "import deep_ep, deep_ep._C as C; print(\"IMPORT_OK\", C.__file__)"
    ' > "$OUT/build.log" 2>&1
    grep -q IMPORT_OK "$OUT/build.log" || { log "build failed, see $OUT/build.log"; exit 3; }
    log "build ok"
fi

run_ep() {  # name stress(0|1) args...
    local name=$1 stress=$2; shift 2
    export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 5000) + ($(date +%s) % 977) ))
    local t0=$(date +%s)
    timeout -k 30 15m srun -N "$N" --ntasks-per-node=1 --mpi=pmix --cpu-bind=none --kill-on-bad-exit=1 \
        --container-image "$SQSH" --container-mounts "$MOUNTS" bash -lc '
            export RANK=${SLURM_PROCID:-0} NCCL_GIN_TYPE=5 NCCL_SYM_GIN_KERNELS_ENABLE=0 NCCL_DEBUG=WARN
            export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 EP_HYBRID_KERNEL=unordered EP_USE_NVIDIA_TOOLS=0 EP_SUPPRESS_NCCL_CHECK=1
            export EP_JIT_EXTRA_FLAGS="-DEP_PROBE_ENTRY_OUTSTANDING=1 -DEP_PROBE_EXIT_FLUSH=1"
            export EP_JIT_CACHE_DIR='"$OUT"'/jit; mkdir -p "$EP_JIT_CACHE_DIR"
            cd '"$TREE"' && export PYTHONPATH='"$TREE"'${PYTHONPATH:+:$PYTHONPATH}
            echo "=== $(hostname) RANK=$RANK WORLD_SIZE=$WORLD_SIZE stress='"$stress"' args=['"$*"'] ==="
            if [ '"$stress"' = 1 ]; then
                torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$(( MASTER_PORT + 7 )) \
                    '"$HERE"'/bg_a2a.py --duration-s '"$BG_DUR"' --report-s 30 > '"$OUT/runs/$name"'_bg_$RANK.txt 2>&1 &
                sleep '"$BG_LEAD"'
            fi
            python3 tests/elastic/test_ep.py '"$*"'
            rc=$?; echo "RUN_EXIT[$(hostname) rank$RANK]=$rc"; exit $rc
        ' > "$OUT/runs/$name.out" 2> "$OUT/runs/$name.err"
    log "run $name rc=$? wall=$(( $(date +%s) - t0 ))s"
}

COMMON="--num-processes=8 --num-topk=6 --num-experts=256 --prefer-overlap-with-compute=0 --skip-check --test-first-only --num-sms=$NUM_SMS"
run_ep quiet-7168  0 $COMMON --num-tokens=8192 --hidden=7168
run_ep quiet-3328  0 $COMMON --num-tokens=4096 --hidden=3328
run_ep stress-7168 1 $COMMON --num-tokens=8192 --hidden=7168
run_ep stress-3328 1 $COMMON --num-tokens=4096 --hidden=3328

for f in "$OUT"/runs/*.out; do
    exits=$(grep -c 'exit flush' "$f")
    pending=$(grep 'exit flush' "$f" | grep -vc 'outstanding before the flush: 0,')
    pct=$(grep -o 'took [0-9]* ns' "$f" | awk '{print $2}' | sort -n | awk '{a[NR]=$1} END{if (NR) printf "p50 %d us, p99 %d us, max %d us", a[int(NR/2)+1]/1000, a[int(NR*0.99)]/1000, a[NR]/1000}')
    echo "== $(basename "$f"): dispatch exits $exits, exits with data WQEs still in flight $pending, flush wait $pct; entries with data outstanding: dispatch $(grep 'probe (hybrid dispatch entry)' "$f" | grep -vc 'data total outstanding: 0$'), combine $(grep 'probe (hybrid combine entry)' "$f" | grep -vc 'data total outstanding: 0$'); errors $(grep -ci 'error\|assert\|mismatch' "$f")"
done | tee "$OUT/summary.txt"
for f in "$OUT"/runs/stress-*_bg_*.txt; do tail -n 2 "$f"; done
log "done -> $OUT"
