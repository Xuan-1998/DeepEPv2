// Device printf FIFO probe: how many records of a timeout line survive when more are written than fit.
//   fifo_probe <blocks> <threads> <records per thread> [fifo_mb] [format: fwd10|fwd14|scaleup9|qstate8|short3|trap]
// The FIFO has no API for its record size, so the size is derived: fifo_bytes / records that survive.
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void spam(int per_thread, int fmt) {
    const int b = blockIdx.x, t = threadIdx.x;
    for (int i = 0; i < per_thread; ++ i) {
        switch (fmt) {
        case 0:  // the forwarding timeout line on main, 10 ints
            printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, channel: %d, lane: %d, "
                   "old scale-out tail: %d, scale-out tail: (%d, %d), arrival anomaly: %d, part: %d, num valid: %d\n",
                   b, t, i, 0, 0, 0, 0, 0, 0, 0);
            break;
        case 1:  // the forwarding timeout line with the completion-missing fields, 14 ints
            printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, channel: %d, lane: %d, "
                   "old scale-out tail: %d, scale-out tail: (%d, %d), arrival anomaly: %d, part: %d, num valid: %d, "
                   "completion missing: %d, part: %d, valid headers: %d, completions: %d\n",
                   b, t, i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            break;
        case 2:  // the combine scale-up wait line, 9 ints
            printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                   "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                   b, 4, t, 8, i, 0, 0, 0, 0);
            break;
        case 3:  // the queue-state line, a string plus 7 ints
            printf("DeepEP GIN queue state (%s), scale-out: %d, scale-up: %d, sm: %d, thread: %d, ctx: %d, %s, "
                   "WQEs not completed by the NIC: %u, written but not rung: %u\n",
                   "hybrid dispatch (forwarding)", b, t, i, 0, 0, "NIC drained", 0u, 0u);
            break;
        default:  // a short line, 3 ints
            printf("TRAPLINE sm %d thread %d record %d\n", b, t, i);
            break;
        }
    }
}

__global__ void spam_then_trap(int per_thread) {
    for (int i = 0; i < per_thread; ++ i)
        printf("TRAPLINE sm %d thread %d record %d\n", (int)blockIdx.x, (int)threadIdx.x, i);
    asm("trap;");
}

int main(int argc, char** argv) {
    const int blocks = atoi(argv[1]), threads = atoi(argv[2]), per = atoi(argv[3]);
    if (argc > 4 && atoi(argv[4]) > 0)
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t)atoi(argv[4]) << 20);
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    const char* name = argc > 5 ? argv[5] : "fwd10";
    fprintf(stderr, "fifo_bytes=%zu records=%d format=%s\n", limit, blocks * threads * per, name);
    if (strcmp(name, "trap") == 0) {
        spam_then_trap<<<blocks, threads>>>(per);
        cudaError_t e = cudaDeviceSynchronize();
        fprintf(stderr, "after trap: %s\n", cudaGetErrorString(e));
        return 0;
    }
    const int fmt = strcmp(name, "fwd14") == 0 ? 1 : strcmp(name, "scaleup9") == 0 ? 2 :
                    strcmp(name, "qstate8") == 0 ? 3 : strcmp(name, "short3") == 0 ? 4 : 0;
    spam<<<blocks, threads>>>(per, fmt);
    cudaDeviceSynchronize();
    return 0;
}
