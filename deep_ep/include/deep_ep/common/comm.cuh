#pragma once

// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cooperative_groups.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/qp_mapping.cuh>

namespace deep_ep::elastic::comm {

static constexpr int64_t kNumOneSecCycles = 2000000000;  // An approximation of the GPU clock at 2000 MHz

// Some reserved tags
static constexpr int kDeviceBarrierTag = 0;
static constexpr int kKernelBarrierTag = 1;
static constexpr int kDispatchTag0 = 2;
static constexpr int kDispatchTag1 = 3;
static constexpr int kCombineTag0 = 4;
static constexpr int kCombineTag1 = 5;
static constexpr int kHybridDispatchTag0 = 6;
static constexpr int kHybridDispatchTag1 = 7;
static constexpr int kHybridCombineTag0 = 8;
static constexpr int kHybridCombineTag1 = 9;

// Some reserved count
static constexpr int kFlushAllAllocatedQPs = -1;

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const bool& condition, const func_t& func,
                                              int64_t start_clock = 0) {
    // User may share a start clock for multiple waits
    if (start_clock == 0)
        start_clock = clock64();

    while (condition) {
        const bool timeout = clock64() - start_clock >= kNumTimeoutCycles;
        if (func(timeout))
            break;

        if (timeout) {
            // Wait another 1 second to let all threads print information and trap
            start_clock = clock64();
            while (clock64() - start_clock < kNumOneSecCycles) {}
            ptx::trap();
        }
    }
}

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const func_t& func, const int64_t& start_clock = 0) {
    timeout_while<kNumTimeoutCycles, func_t>(true, func, start_clock);
}

// Watchdog for loops without a single condition to wait on (the combine proxy warp sweeps several rings):
// the caller says whether the iteration made progress, and after two timeout budgets without any, `func`
// prints and the thread traps. Two budgets, so that a forward warp waiting for its own data reports first.
template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void check_progress(int64_t& last_progress_clock, const bool& progressed,
                                               const func_t& func) {
    const auto now = clock64();
    if (progressed) {
        last_progress_clock = now;
        return;
    }
    if (now - last_progress_clock < 2 * kNumTimeoutCycles)
        return;
    func();
    while (clock64() - now < kNumOneSecCycles) {}
    ptx::trap();
}

__device__ __forceinline__ void print_proxy_ring_state(const char* where, const int scaleout_rank_idx,
                                                       const int scaleup_rank_idx, const int sm_idx,
                                                       const int forward_warp_idx, const int ring_done,
                                                       const unsigned head, const unsigned tail, const int done_flag) {
    printf("DeepEP combine (%s) no progress timeout, scale-out: %d, scale-up: %d, sm: %d, forward warp: %d, "
           "ring done: %d, head: %u, tail: %u, done flag: %d\n",
           where, scaleout_rank_idx, scaleup_rank_idx, sm_idx, forward_warp_idx, ring_done, head, tail, done_flag);
}

// `ncclGin::put` and `ncclGin::signal` spin without a timeout in the EFA GDA backend while the send queue is
// full: the slot to reserve must be within `sq_size` of the NIC's FI_WRITE counter. The posting lane waits
// here first, with a timeout, for room for `num_wqes` more WQEs. The NIC counter only grows, so the caller's
// `cached_completed` is a lower bound of it: the fast path is one load of the reservation cursor, and the
// counter is read about once per `sq_size` posts.
static constexpr uint32_t kSqCompletedUnset = 0xffffffffu;  // Never a counter value, the counter wraps at 2^31

template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ uint32_t wait_sq_room_slow(const char* where, const ncclGinCtx& ctx, const uint32_t& num_wqes,
                                                      const uint32_t& reserved,
                                                      const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
    uint32_t completed = 0;
#if NCCL_GIN_EFA_GDA_ENABLE
    const auto* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    const uint32_t sq_size = dev->data.sq_size;
    timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
        completed = static_cast<uint32_t>(
            nccl::gin::efa_gda::hwCounterLoad<cuda::memory_order_relaxed>(dev->data.local_cntr_value));
        if (((reserved + num_wqes - completed) & nccl::gin::efa_gda::EFA_CNTR_MASK) <= sq_size)
            return true;
        if (is_last_check) {
            printf("DeepEP GIN send queue full timeout (%s), scale-out: %d, scale-up: %d, sm: %d, thread: %d, "
                   "ctx: %d, SQ size: %u, reserved: %u, completed by the NIC: %u, wanted: %u WQEs\n",
                   where, scaleout_rank_idx, scaleup_rank_idx, static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x),
                   ctx.contextId, sq_size, reserved, completed, num_wqes);
        }
        return false;
    });
#endif
    return completed;
}

template <int64_t kNumTimeoutCycles, typename nccl_gin_t>
__device__ __forceinline__ void wait_sq_room(const char* where, const nccl_gin_t& nccl_gin, const uint32_t& num_wqes,
                                             uint32_t& cached_completed,
                                             const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
#if NCCL_GIN_EFA_GDA_ENABLE
    const auto ctx = nccl_gin._makeCtx();
    if (ctx.backend != NCCL_NET_DEVICE_GIN_EFA_GDA)
        return;
    const auto* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    const auto reserved = *reinterpret_cast<const volatile uint32_t*>(
        &static_cast<const efa_cuda_qp*>(dev->data.qp)->sq.wq.pc);
    if (cached_completed != kSqCompletedUnset and
        ((reserved + num_wqes - cached_completed) & nccl::gin::efa_gda::EFA_CNTR_MASK) <= dev->data.sq_size)
        return;
    cached_completed = wait_sq_room_slow<kNumTimeoutCycles>(where, ctx, num_wqes, reserved,
                                                            scaleout_rank_idx, scaleup_rank_idx);
#endif
}

// Shared memory of the diagnostics below: the last `kNumDiagSmemBytes` of the dynamic region, which the host
// leaves out of every shared-memory layout. Static `__shared__` would not fit, as the kernels launch with the
// whole opt-in size. A kernel launched with less dynamic memory (the barrier kernel launches with none) gets
// `nullptr`.
static constexpr int kDiagNannyMaskOffset = 0;          // `uint32_t[32]`, one per warp
static constexpr int kDiagProxyPutStartOffset = 128;    // `int64_t[32]`, one per lane of the proxy warp
static constexpr int kDiagProxyActiveOffset = 384;      // `int`
EP_STATIC_ASSERT(kDiagProxyActiveOffset + static_cast<int>(sizeof(int)) <= kNumDiagSmemBytes, "Diagnostics exceed their reserve");

__device__ __forceinline__ int8_t* diag_smem_base() {
    extern __shared__ __align__(16) int8_t diag_smem_alias[];
    uint32_t num_dynamic_smem_bytes;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(num_dynamic_smem_bytes));
    if (num_dynamic_smem_bytes < static_cast<uint32_t>(kNumDiagSmemBytes))
        return nullptr;
    return diag_smem_alias + num_dynamic_smem_bytes - kNumDiagSmemBytes;
}

// Lane nanny for GIN posts. Besides a full send queue, `ncclGin::put` can spin forever on the backend's
// un-rung backlog or on the doorbell turn of an earlier poster, which `wait_sq_room` cannot see. Call from
// all 32 lanes: the lanes with `do_post` run `post`, the lowest other lane watches a per-warp mask of the
// lanes that returned and reports the ones that did not. Independent thread scheduling runs the two
// divergent paths side by side. There is no nanny when all 32 lanes post or without diagnostics memory.
template <int64_t kNumTimeoutCycles, typename post_t>
__device__ __forceinline__ void guarded_post(const bool& do_post, const post_t& post, const char* where,
                                             const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
    const unsigned post_mask = __ballot_sync(0xffffffffu, do_post);
    if (post_mask == 0)
        return;

    // NOTES: `post` must be inlined only once, a second copy of `ncclGin::put` costs spills in the dispatch
    const int warp_idx = static_cast<int>(threadIdx.x) / 32, lane_idx = static_cast<int>(threadIdx.x) % 32;
    const auto diag_smem = diag_smem_base();
    const bool has_nanny = diag_smem != nullptr and post_mask != 0xffffffffu;
    const int nanny_lane_idx = has_nanny ? __ffs(~post_mask) - 1 : -1;
    const auto returned = has_nanny ? reinterpret_cast<unsigned*>(diag_smem + kDiagNannyMaskOffset) + warp_idx : nullptr;
    if (lane_idx == nanny_lane_idx)
        *returned = 0;
    __syncwarp();

    if (do_post) {
        post();
        if (has_nanny) {
            __threadfence_block();
            atomicOr_block(returned, 1u << lane_idx);
        }
    } else if (lane_idx == nanny_lane_idx) {
        timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            const auto returned_mask = *reinterpret_cast<volatile unsigned*>(returned);
            if (returned_mask == post_mask)
                return true;
            if (is_last_check) {
                printf("DeepEP GIN post stuck (%s), scale-out: %d, scale-up: %d, sm: %d, warp: %d, "
                       "stuck lanes: 0x%08x, posting lanes: 0x%08x\n",
                       where, scaleout_rank_idx, scaleup_rank_idx, static_cast<int>(blockIdx.x), warp_idx,
                       post_mask & ~returned_mask, post_mask);
            }
            return false;
        });
    }
    __syncwarp();
}

// Watchdog lane for the combine proxy warp, whose posting lanes run their own loops, so a warp-uniform
// nanny does not fit. Each posting lane stamps the start of a put and clears it after, and the watching
// lane scans the stamps until the active count of posting lanes reaches 0.
__device__ __forceinline__ volatile int64_t* proxy_put_start_slots() {
    return reinterpret_cast<volatile int64_t*>(diag_smem_base() + kDiagProxyPutStartOffset);
}

__device__ __forceinline__ int* proxy_active_slot() {
    return reinterpret_cast<int*>(diag_smem_base() + kDiagProxyActiveOffset);
}

template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void proxy_watchdog(const int& num_posting_lanes, const char* where,
                                               const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
    const auto active = reinterpret_cast<volatile int*>(proxy_active_slot());
    const auto put_start = proxy_put_start_slots();
    while (*active > 0) {
        const auto now = clock64();
        for (int i = 0; i < num_posting_lanes; ++ i) {
            const auto start = put_start[i];
            if (start != 0 and now - start >= kNumTimeoutCycles) {
                printf("DeepEP GIN post stuck (%s), scale-out: %d, scale-up: %d, sm: %d, lane: %d inside put for %lld cycles\n",
                       where, scaleout_rank_idx, scaleup_rank_idx, static_cast<int>(blockIdx.x), i,
                       static_cast<long long>(now - start));
                const auto print_clock = clock64();
                while (clock64() - print_clock < kNumOneSecCycles) {}
                ptx::trap();
            }
        }
        __nanosleep(2000);
    }
}

template <int64_t kNumTimeoutCycles, typename gin_t, typename coop_t>
__device__ __forceinline__ void flush_with_timeout(const gin_t& gin, const coop_t& coop,
                                                   const int& scaleout_rank_idx = -1, const int& scaleup_rank_idx = -1) {
    if (gin.flush(coop, cuda::memory_order_acquire, ncclGin_None{},
                  static_cast<uint64_t>(kNumTimeoutCycles)) != ncclSuccess) {
        printf("DeepEP Gin flush timeout, scale-out: %d, scale-up: %d, sm: %d, thread: %d\n",
               scaleout_rank_idx, scaleup_rank_idx, static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x));
        ptx::trap();
    }
}

// Channels map onto QPs with a BALANCED CONTIGUOUS (block) partition: fill a QP with
// consecutive channels before spilling to the next (so an SM's channels stay grouped
// on the same GIN context), and when the channels do not divide the QPs evenly,
// spread the remainder one-per-QP across the leading QPs instead of leaving a
// trailing QP idle. Used by all callers (dispatch, hybrid_dispatch, combine,
// hybrid_combine). The integer partition lives in `qp_mapping.cuh` (host-testable);
// this wrapper only adds the NCCL resource-sharing mode.
//   e.g. 48 channels / 4 QPs  -> {12,12,12,12}  (even; identical to before)
//        4  channels / 3 QPs  -> QP0<-2, QP1<-1, QP2<-1  (was {2,2,0}, QP2 idle)
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp = false) {
    static_assert(kNumQPs >= 1,
                  "kNumQPs must be >= 1. With 0 the kNumQPs == 1 fast path is skipped, "
                  "kNumSMs <= kNumAvailableQPs is false, and balanced_partition() is called "
                  "with q == 0, i.e. n / 0. Compiled for the host this specialization exits on "
                  "SIGFPE; in a GPU run it produced corrupted output and no reported fault.");

    constexpr auto kSharingCTA = NCCL_GIN_RESOURCE_SHARING_CTA;
    constexpr auto kSharingGrid = kNumSMs == 1 ? NCCL_GIN_RESOURCE_SHARING_CTA : NCCL_GIN_RESOURCE_SHARING_GPU;

    // Only one QP
    if constexpr (kNumQPs == 1)
        return {0, kSharingGrid};

    // The notify warp always use 1 SM and 1 QP
    if (is_notify_warp)
        return {0, kSharingCTA};

    // Data channels: QP index from the shared balanced partition, sharing mode from
    // the branch (CTA when SMs each own their QPs, GPU-wide when SMs share QPs).
    constexpr int kNumAvailableQPs = kNumQPs - static_cast<int>(kWithNotifyWarps);
    const int qp_idx = channel_to_qp<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
        sm_idx, channel_in_sm_idx, is_notify_warp);
    if constexpr (kNumSMs <= kNumAvailableQPs)
        return {qp_idx, kSharingCTA};
    else
        return {qp_idx, kSharingGrid};
}

// Companion to `get_qp_mode`: within a QP, gives a unique per-channel signal id
// (position of this channel among all channels sharing the same QP). Uses the same
// BALANCED CONTIGUOUS partition as `get_qp_mode` (see `qp_mapping.cuh`), so the id is
// the channel's offset within its QP's block:
//   kNumQPs == 1                : every channel on the single QP -> id = global_channel_idx
//   kNumSMs <= kNumAvailableQPs : offset within the SM's balanced local-QP block
//   kNumSMs  > kNumAvailableQPs : offset within the global balanced QP block
// In all cases id < ceil(channels / qps), so `(qp, id)` is unique per channel and the
// tuner's per-QP signal budget (sized for ceil(channels / qps)) is never exceeded.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_qp_signal_id(
    const int& sm_idx, const int& channel_in_sm_idx) {
    return channel_to_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
        sm_idx, channel_in_sm_idx);
}

// Per-part indexed-signal id: kNumParts contiguous ids under the channel's base id, one
// per token part and shared by all sources. The tuner + channel cap guarantee
// ceil(num_channels / qps) * kNumParts <= gin_indexed_signals_cnt.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, int kNumParts,
          bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_per_part_signal_id(
    const int& sm_idx, const int& channel_in_sm_idx, const int& part_idx) {
    return get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
               sm_idx, channel_in_sm_idx) * kNumParts + part_idx;
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag>
__forceinline__ __device__ void nvlink_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const layout::WorkspaceLayout& workspace,
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {
    // This barrier only uses 1 SM
    if (kNumSMs > 1 and sm_idx > 0)
        return;

    // Read the current barrier phase first
    const int status = static_cast<int>((*workspace.get_nvl_barrier_counter_ptr()) & 3);
    const int phase = status & 1, sign = status >> 1;

    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (thread_idx < kNumRanks) {
        const auto dst_ptr =
            gin.get_sym_ptr<ncclTeamTagLsa>(workspace.get_nvl_barrier_signal_ptr(phase), thread_idx);
        ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);
    }
    __syncthreads();

    // NOTES: we need `2^64 / 1e6 / 3600 / 24 / 365 = 571000` years to make the counter overflow (1 barrier per us)
    // Add the phase counter
    if (thread_idx == 0)
        atomicAdd(workspace.get_nvl_barrier_counter_ptr(), 1);

    // Check timeout
    const auto target = sign ? 0 : kNumRanks;
    timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=](const bool& is_last_check) {
        const auto signal = ptx::ld_acquire_sys<int>(workspace.get_nvl_barrier_signal_ptr(phase));
        if (signal == target)
            return true;

        if (is_last_check) {
            printf("DeepEP NVLink barrier timeout, tag: %d, nvl: %d, thread: %d, "
                   "status: %d, signal: %d, phase: %d, target: %d, counter: %llu\n",
                   kTag, rank_idx, thread_idx, status, signal, phase, target,
                   *workspace.get_nvl_barrier_counter_ptr());
        }
        return false;
    });
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles,
          typename team_t, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true,
          int kNumWarps = kNumThreads / 32>
__forceinline__ __device__ void gin_barrier_wo_local_sync(
    const ncclDevComm_t& nccl_dev_comm,
    const int& scaleout_rank_idx, const int& scaleup_rank_idx, 
    const int& sm_idx, const int& thread_idx) {
    const auto global_warp_idx = sm_idx * kNumWarps + (thread_idx / 32);
    const int& rank_idx = (std::is_same_v<team_t, ncclTeamTagWorld>) ? scaleup_rank_idx : scaleout_rank_idx;
    const int num_qps = kNumQPs == kFlushAllAllocatedQPs ? nccl_dev_comm.ginContextCount : kNumQPs;

    // Flush all QPs by all SMs (only needed for release semantics)
    if constexpr (kFlushStores) {
        for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {
            flush_with_timeout<kNumTimeoutCycles>(ncclGin(nccl_dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA),
                                                  ncclCoopWarp(), scaleout_rank_idx, scaleup_rank_idx);
        }

        // World-team barriers may mix direct NVLink writes with GIN signals, so a system-scope
        // threadfence is required to make the NVLink writes visible before publishing the barrier
        if constexpr (std::is_same_v<team_t, ncclTeamTagWorld>)
            ptx::fence_acq_rel_sys();

        // NOTES: we can not use `kNumSMs` to judge, as maybe only part of the SMs will call this function
        (gridDim.x > 1) ? cooperative_groups::this_grid().sync() : __syncthreads();
    }

    if (sm_idx == 0) {
        // Use QP 0 to do barrier
        const auto team = (std::is_same_v<team_t, ncclTeamTagWorld>) ?
            ncclTeamWorld(nccl_dev_comm) : ncclTeamRail(nccl_dev_comm);
        const ncclGin gin(nccl_dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

        // Compact signal indexing: (kNumRanks - 1) signal slots per rank. Sender rank_idx
        // writes to every peer i at the slot that identifies *itself* in the peer's
        // enumeration:
        //   sig = (rank_idx < i) ? rank_idx : (rank_idx - 1)
        // So on receiver R, each of the (kNumRanks - 1) slots gets exactly +1 from a
        // distinct sender, and the wait side just iterates all slots looking for one
        // increment per slot.
        // NOTES: every thread runs the same trip count, `guarded_post` needs warp-uniform calls
        uint32_t sq_completed_cache = kSqCompletedUnset;
        for (int base = 0; base < kNumRanks; base += kNumThreads) {
            const int i = base + thread_idx;
            guarded_post<kNumTimeoutCycles>(i < kNumRanks and i != rank_idx, [&]() {
                const auto sig = static_cast<ncclGinSignal_t>((rank_idx < i) ? rank_idx : (rank_idx - 1));
                wait_sq_room<kNumTimeoutCycles>("barrier signal", gin, 1, sq_completed_cache, scaleout_rank_idx, scaleup_rank_idx);
                gin.signal(team, i, ncclGin_SignalInc{sig});
            }, "barrier signal", scaleout_rank_idx, scaleup_rank_idx);
        }

        for (int i = thread_idx; i < kNumRanks - 1; i += kNumThreads) {
            const auto signal_idx = static_cast<ncclGinSignal_t>(i);
            const auto shadow_ptr = gin.getSignalShadowPtr(signal_idx);
            const auto target = ++(*shadow_ptr);

            // TODO(NCCL): Using the official NCCL wait signal API, after they added timeout check.
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = gin.readSignal(signal_idx, 64, cuda::memory_order_acquire);
                if (signal >= target)
                    return true;

                if (is_last_check) {
                    printf("DeepEP Gin barrier timeout, tag: %d, scaleout: %d, scaleup: %d, thread: %d, "
                           "signal: %lu, target: %lu\n", kTag, scaleout_rank_idx, scaleup_rank_idx, thread_idx, signal, target);
                }
                return false;
            });
        }
    }
}

template <bool kIsScaleupNVLink, int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag, bool kFlushStores = true>
__forceinline__ __device__ void scaleup_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const layout::WorkspaceLayout& workspace,
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {
    if constexpr (kIsScaleupNVLink) {
        nvlink_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumTimeoutCycles, kTag>(
            gin, workspace, rank_idx, sm_idx, thread_idx);
    } else {
        gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagWorld, kTag, kFlushStores>(
            gin.nccl_dev_comm, 1, rank_idx, sm_idx, thread_idx);
    }
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true>
__forceinline__ __device__ void scaleout_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const int& scaleout_rank_idx, const int& scaleup_rank_idx,
    const int& sm_idx, const int& thread_idx) {
    gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagRail, kTag, kFlushStores>(
        gin.nccl_dev_comm, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
}

template <bool kIsScaleupNVLink,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true>
__forceinline__ __device__ void gpu_barrier(const handle::NCCLGin& gin,
                                            const layout::WorkspaceLayout& workspace,
                                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                            const int& sm_idx, const int& thread_idx,
                                            bool do_scaleout = true, bool do_scaleup = true) {
    // A general TMA store wait to prevent proxy memory issues
    if constexpr (kFlushStores) {
        ptx::tma_store_commit();
        ptx::tma_store_wait();
        __syncwarp();
    }

    // All the SMs should wait
    if constexpr (kSyncAtStart) {
        cooperative_groups::this_grid().sync();
    } else {
        EP_STATIC_ASSERT(not kFlushStores, "No data to be flushed");
    }

    do_scaleout &= kNumScaleoutRanks > 1;
    do_scaleup &= kNumScaleupRanks > 1;
    if (do_scaleup and do_scaleout) {
        // Do scaleup and scaleout barrier in parallel
        EP_DEVICE_ASSERT(kNumSMs >= 2 and "At least 2 SMs for a hybrid barrier");
        if (sm_idx == 0) {
            // First SM do the scaleup barrier
            scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);

            // We need an extra grid sync, as the scaleout barrier will do a sync after flush, before the barrier
            // NOTES: this is kind of hacky
            if constexpr (kFlushStores) 
                cooperative_groups::this_grid().sync();
        } else {
            // The remaining SMs do the scaleout barrier
            scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs - 1, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx - 1, thread_idx);
        }
    } else if (do_scaleup) {
        // Scaleup only
        scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);
    } else if (do_scaleout) {
        // Scaleout only
        scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }

    // All the SMs should wait
    if constexpr (kSyncAtEnd)
        cooperative_groups::this_grid().sync();
}

}  // namespace deep_ep::elastic::comm
