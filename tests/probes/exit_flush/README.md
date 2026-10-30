# Dispatch-exit in-flight probe and EFA burst stressor

Two compile-time probes of the unordered hybrid kernels (EFA GDA backend only), switched on with
`EP_JIT_EXTRA_FLAGS="-DEP_PROBE_ENTRY_OUTSTANDING=1 -DEP_PROBE_EXIT_FLUSH=1"`, and the background
traffic that was used to stress the fabric while they ran. The kernels are unchanged without the flags.

Both probes read the EFA GDA device handle of each GIN context: `submitted_count` (work requests whose
doorbell was rung) and the NIC-written counter `local_cntr_value` (work requests completed locally).
Their difference, masked to 31 bits, is what the backend's own `flush` waits on.

- `EP_PROBE_ENTRY_OUTSTANDING`: after the entry barrier of the dispatch and of the combine, thread 0 of SM 0
  prints one line per kernel per GPU:
  `DeepEP probe (hybrid dispatch entry), ..., barrier ctx outstanding: B, data ctxs with outstanding WQEs: C,
  data max outstanding: M, data total outstanding: T`. Context 0 carries the barrier signals the entry barrier
  just posted, so it is reported apart (`B`). `T > 0` at a combine entry means this GPU's NIC still held dispatch
  puts when the combine started.
- `EP_PROBE_EXIT_FLUSH`: after the NVLink exit barrier of the dispatch, SM 0 counts the data work requests the NIC
  has not completed, then flushes every context and times the flush with `%globaltimer`:
  `DeepEP probe (hybrid dispatch exit flush), ..., data WQEs outstanding before the flush: W, flush of K contexts
  took N ns`. `W > 0` is a dispatch exit with puts still in flight, and `N` is what a `gin.flush()` at that point
  waits for.

On a tree where the sender warps already flush before the exit barrier (`comm::flush_with_timeout` in
`hybrid_dispatch_unordered.cuh`), the exit probe reads what that flush left behind, which should be 0. To measure
the exit without that flush, remove those calls (one commit on `main`) and run the probe again.

`bg_a2a.py` is the stressor: incast and outcast all-to-all bursts, a uniform all-to-all and four concurrent
all-reduces over all ranks, in a loop, until `--duration-s`. Local rank 0 of each node prints the node's EFA
counter deltas every `--report-s` seconds (`tx_pkts`, `retrans_pkts` per million, `retrans_timeout_events`), so a
run log shows whether the fabric did retransmit while the kernel under test ran. Start it with `torchrun` in the
same step as the test, before the test (see the runner).

`run-probe.sbatch` runs `tests/elastic/test_ep.py` in perf mode (no checks) with both probes, for two shapes,
quiet and stressed, and prints per run: dispatch exits, exits with data work requests still in flight, the flush
wait p50/p99/max, entries with data outstanding, and the stressor's last counter lines.

```
TREE=/path/to/this/checkout SQSH=/path/to/image.sqsh sbatch -N 4 --job-name=<name> tests/probes/exit_flush/run-probe.sbatch
```
