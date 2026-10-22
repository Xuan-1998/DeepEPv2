// Generated from the timeout printf formats of the branch: records that survive per format.
//   fifo_probe_all <fifo_mb> <format index>   (writes 44,800 lines of one format; --names lists the formats)
#include <cstdio>
#include <cstdlib>

__global__ void spam(int per_thread, int fmt) {
    for (int i = 0; i < per_thread; ++ i) {
        switch (fmt) {
        case 0: printf("DeepEP GIN queue state (%s), scale-out: %d, scale-up: %d, sm: %d, thread: %d, ctx: %d, %s, WQEs not completed by the NIC: %u, written but not rung: %u\n", "x", 0, 0, 0, 0, 0, "x", 0u, 0u); break;
        case 1: printf("DeepEP GIN queue state (%s), scale-out: %d, scale-up: %d, sm: %d, thread: %d, ctx: %d, unavailable on backend %d\n", "x", 0, 0, 0, 0, 0, 0); break;
        case 2: printf("DeepEP Gin flush timeout, sm: %d, thread: %d\n", 0, 0); break;
        case 3: printf("DeepEP grid sync timeout (%s), scale-out: %d, scale-up: %d, sm: %d, SMs arrived: %u of %u (the SM without this line did not arrive)\n", "x", 0, 0, 0, 0u, 0u); break;
        case 4: printf("DeepEP NVLink barrier timeout, tag: %d, nvl: %d, thread: %d, status: %d, signal: %d, phase: %d, target: %d, counter: %llu\n", 0, 0, 0, 0, 0, 0, 0, 0ull); break;
        case 5: printf("DeepEP Gin barrier timeout, tag: %d, scaleout: %d, scaleup: %d, thread: %d, signal: %lu, target: %lu\n", 0, 0, 0, 0, 0ul, 0ul); break;
        case 6: printf("DeepEP hybrid notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, thread: %d, status: %d | %d, expected: %d\n", 0, 0, 0, 0, 0, 0, 0, 0); break;
        case 7: printf("DeepEP hybrid notify (scale-out %s reduction) timeout, scale-out: %d, scale-up: %d, thread: %d, wait scale-out: %d, decoded: %d\n", "x", 0, 0, 0, 0, 0); break;
        case 8: printf("DeepEP hybrid notify (scale-up reduction) timeout,scale-out: %d/%d, scale-up: %d/%d, thread: %d, status: %d | %d, expected: %d\n", 0, 0, 0, 0, 0, 0, 0, 0); break;
        case 9: printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d), arrival anomaly: %d, part: %d, num valid: %d, completion missing: %d, part: %d, valid headers: %d, completions: %d\n", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
        case 10: printf("DeepEP hybrid dispatch (epilogue shadow drain) timeout, scale-out: %d, scale-up: %d, channel: %d, part: %d, landed: %lld, expected: %d\n", 0, 0, 0, 0, 0ll, 0); break;
        case 11: printf("DeepEP combine (proxy ring full) timeout, scale-out: %d, scale-up: %d, channel: %d, forward warp: %d, dst: %d, head: %u, tail: %u, depth: %d\n", 0, 0, 0, 0, 0, 0u, 0u, 0); break;
        case 12: printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n", 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
        case 13: printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, channel: %d, signal: %lu, target: %lu\n", 0, 0, 0, 0, 0, 0ul, 0ul); break;
        }
    }
}

int main(int argc, char** argv) {
    // fifo_probe_all <fifo_mb> <format index>: one format per process, so the stdout line count is this format's
    // fifo_probe_all --names: list the format names
    const char* names[] = {"GIN_queue_state_s", "GIN_queue_state_s", "Gin_flush_timeout", "grid_sync_timeout_s", "NVLink_barrier_timeout", "Gin_barrier_timeout", "hybrid_notify_GPU_reduction_timeout", "hybrid_notify_scale_out_s_reduction_time", "hybrid_notify_scale_up_reduction_timeout", "hybrid_dispatch_forwarding_timeout", "hybrid_dispatch_epilogue_shadow_drain_ti", "combine_proxy_ring_full_timeout", "combine_scale_up_wait_timeout", "combine_scale_out_wait_all_timeout"};
    if (argc > 1 && argv[1][0] == '-') {
        for (int f = 0; f < 14; ++ f) printf("%d %s\n", f, names[f]);
        return 0;
    }
    if (argc > 1 && atoi(argv[1]) > 0) cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t)atoi(argv[1]) << 20);
    size_t limit = 0; cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    const int f = argc > 2 ? atoi(argv[2]) : 0;
    fprintf(stderr, "fifo_probe_all %s fifo_bytes=%zu records=44800", names[f], limit);
    spam<<<28, 32>>>(50, f);
    cudaDeviceSynchronize();
    return 0;
}
