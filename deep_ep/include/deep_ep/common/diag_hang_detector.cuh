#pragma once

// MIT License
//
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// The `hang_detector` check of `EP_DEBUG_CHECKS`. The bit is in `compiled.cuh`, see `diag.cuh`

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/diag_gin.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::diag::hang_detector {

constexpr bool kEnabled = (kDebugChecks & kDebugCheckHangDetector) != 0;

// With the check on, an idle lane of the warp watches the lanes that post, since `ncclGin::put` has no timeout.
// The words live in the last `kNumHangDetectorSmemBytes` of the dynamic shared memory, outside the TMA pool.
__device__ __forceinline__ unsigned* words() {
    extern __shared__ __align__(16) int8_t hang_detector_smem[];
    uint32_t num_smem_bytes;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(num_smem_bytes));
    return reinterpret_cast<unsigned*>(hang_detector_smem + (num_smem_bytes - kNumHangDetectorSmemBytes));
}

// The lowest lane that does not post, or -1 when every lane posts
__device__ __forceinline__ int watcher_lane(const unsigned& poster_mask) {
    return ptx::ffs(~poster_mask);
}

__device__ __noinline__ void report(const unsigned stuck_mask, const char* where,
                                    const int scaleout_rank_idx, const int scaleup_rank_idx) {
    printf("DeepEP GIN post stuck (%s), scale-out: %d, scale-up: %d, sm: %d, warp: %d, stuck lanes: 0x%08x\n",
           where, scaleout_rank_idx, scaleup_rank_idx, static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x / 32), stuck_mask);
    trap_after_report();
}

// For a converged warp. The posting lanes run `post`, and the lowest idle lane waits for them in `timeout_while`
// and prints the lanes that did not return. One word per warp of the CTA, so every warp of the kernel may call it.
template <int64_t kNumTimeoutCycles, int kNumThreads, typename post_t>
__device__ __forceinline__ void guarded_post(const bool& do_post, const post_t& post, const char* where,
                                             const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
    EP_STATIC_ASSERT(kNumThreads / 32 <= kNumHangDetectorWords, "One word per warp of the CTA");
    if constexpr (not kEnabled) {
        if (do_post)
            post();
        return;
    }
    const unsigned post_mask = ptx::gather(do_post);
    if (post_mask == 0)
        return;
    auto* returned = words() + threadIdx.x / 32;
    const int lane_idx = static_cast<int>(threadIdx.x) % 32;
    const int watcher_lane_idx = watcher_lane(post_mask);
    if (lane_idx == watcher_lane_idx)
        *returned = 0;
    __syncwarp();
    if (do_post) {
        post();
        atomicOr_block(returned, 1u << lane_idx);
    } else if (lane_idx == watcher_lane_idx) {
        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            const auto returned_mask = *reinterpret_cast<volatile unsigned*>(returned);
            if (returned_mask == post_mask)
                return true;
            if (is_last_check)
                report(post_mask & ~returned_mask, where, scaleout_rank_idx, scaleup_rank_idx);
            return false;
        });
    }
    __syncwarp();
}

// For a warp whose lanes post on their own schedule (the combine proxy). Each posting lane stamps the clock into its
// word for the duration of its put, and the lowest idle lane watches the stamps until every posting lane has called
// `done`. One word per lane, so only `kNumLaneWarps` warps of the kernel may call these, the proxy warp alone in
// the combine. The watcher's own word holds the done mask.
static constexpr int kStampShift = 11;
static constexpr unsigned kWatchPollNanos = 2000;

__device__ __forceinline__ unsigned stamp() {
    return static_cast<unsigned>(clock64() >> kStampShift) | 1u;
}

template <int kNumLaneWarps>
__device__ __forceinline__ void init() {
    EP_STATIC_ASSERT(kNumLaneWarps * 32 <= kNumHangDetectorWords, "One word per lane of the warps that index by lane");
    if constexpr (kEnabled) {
        words()[threadIdx.x % 32] = 0;
        __syncwarp();
    }
}

__device__ __forceinline__ void post_begin() {
    if constexpr (kEnabled)
        reinterpret_cast<volatile unsigned*>(words())[threadIdx.x % 32] = stamp();
}

__device__ __forceinline__ void post_end() {
    if constexpr (kEnabled)
        reinterpret_cast<volatile unsigned*>(words())[threadIdx.x % 32] = 0;
}

__device__ __forceinline__ void done(const unsigned& poster_mask) {
    if constexpr (kEnabled)
        atomicOr_block(words() + watcher_lane(poster_mask), 1u << (threadIdx.x % 32));
}

template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void watch(const unsigned& poster_mask, const char* where,
                                      const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
    if constexpr (kEnabled) {
        const int lane_idx = static_cast<int>(threadIdx.x) % 32;
        const int watcher_lane_idx = watcher_lane(poster_mask);
        const bool every_lane_posts = poster_mask == 0xffffffffu;
        if (every_lane_posts or lane_idx != watcher_lane_idx)
            return;
        constexpr int kNumTimeoutStamps = static_cast<int>(kNumTimeoutCycles >> kStampShift);
        auto* stamps = reinterpret_cast<volatile unsigned*>(words());
        unsigned stuck_mask = 0;
        while (stuck_mask == 0 and (stamps[watcher_lane_idx] & poster_mask) != poster_mask) {
            const auto now = stamp();
            #pragma unroll 1
            for (auto pending = poster_mask; pending != 0; pending &= pending - 1) {
                const int lane = ptx::ffs(pending);
                const auto lane_stamp = stamps[lane];
                if (lane_stamp != 0 and static_cast<int>(now - lane_stamp) > kNumTimeoutStamps)
                    stuck_mask |= 1u << lane;
            }
            __nanosleep(kWatchPollNanos);
        }
        if (stuck_mask != 0)
            report(stuck_mask, where, scaleout_rank_idx, scaleup_rank_idx);
    }
}

}  // namespace deep_ep::elastic::diag::hang_detector
