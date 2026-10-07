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

// GIN resources of the unordered hybrid kernels: the context (QP) and per-context
// indexed-signal budget, the channel to QP mapping and the per-channel signal ids.

#include <algorithm>
#include <utility>

#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

namespace deep_ep::ep::unordered {

// Provider resource budget: a GIN context costs one data QP, one QP for value puts and one
// QP per indexed signal, out of 256 QPs per device communicator
static constexpr int kTotalQPBudget = 256;
static constexpr int kMaxGinContextBudget = 17;

// Scale-out warps per SM assumed by the budget math. Matches the buffer's cap on
// `num_channels_per_sm` when `prefer_overlap_with_compute` is off (one scale-out warp per
// channel). The runtime peak can reach `kNumMaxChannelsPerSM = 8`, which only means heavier
// warps-per-context sharing, not a correctness issue.
static constexpr int kMaxWarpsPerSM = 4;

// Upper bound on scale-out warps used by any single kernel: 55 SMs x 4 warps = 220.
// Each scale-out warp (channel) needs its own signal id, reused for that channel across all
// peers of the rail team, so the signal count scales with the channel count and not with the
// number of nodes.
static constexpr int kMaxSM = (kTotalQPBudget - 2 * kMaxGinContextBudget) / kMaxWarpsPerSM;
static constexpr int kMaxScaleoutWarps = kMaxSM * kMaxWarpsPerSM;

struct GinResourceConfig {
    int gin_indexed_signals_cnt;      // Indexed signals per GIN context
    int gin_context_cnt;              // Scale-out contexts plus one notify context
};

static constexpr int kMinGinContextCnt = 2;
static constexpr int kMaxGinContextCnt = kMaxGinContextBudget;

// Default context count (and QP count). 11 contexts give 21 signals per context. Contexts and
// signals per context are inversely coupled through `gin_indexed_signals_for`, so more QPs
// means a smaller per-context signal budget. {5, 6, 7, 8, 9, 14} are equivalent, every other
// value loses a part somewhere: 12, 15, 16 and 17 all drop to 3 parts at 12 SMs, and 17 (the
// provider maximum) leaves only 13 signals per context with 4 channels on the busiest QP.
static constexpr int kDefaultGinContextCnt = 11;

__forceinline__ __device__ __host__ constexpr int gin_indexed_signals_for(int gin_context_cnt) {
    return (kTotalQPBudget - 2 * gin_context_cnt) / gin_context_cnt;
}

__forceinline__ __device__ __host__ constexpr GinResourceConfig make_gin_resources(int gin_context_cnt) {
    return GinResourceConfig{gin_indexed_signals_for(gin_context_cnt), gin_context_cnt};
}

// The total signal budget (contexts x signals per context) must cover the worst-case warp
// count for every legal context count, not just the default. The tightest points are 13 and
// 17 contexts, both at 221 against the 220-warp ceiling. Re-check before raising `kMaxSM` or
// `kMaxWarpsPerSM`, widening the context range, or lowering `kTotalQPBudget`.
__forceinline__ __host__ constexpr bool all_gin_context_counts_cover_warps() {
    for (int ctx = kMinGinContextCnt; ctx <= kMaxGinContextCnt; ++ ctx)
        if (ctx * gin_indexed_signals_for(ctx) < kMaxScaleoutWarps)
            return false;
    return true;
}
static_assert(all_gin_context_counts_cover_warps(),
              "GIN layout cannot give each scale-out warp a dedicated signal id "
              "for every legal context count");

// Maximum number of parts a channel's tokens are split into, one indexed signal per part
static constexpr int kMaxParts = 4;

// Extra slots per channel so that every part rounds up to whole slots
static constexpr int kScaleoutSlotRoundingReserve = kMaxParts;

struct GinPartAllocation {
    int num_parts;           // per-channel part count (>= 1)
    int num_channels_per_sm;
};

// Worst-case number of channels that land on a single GIN context (QP). Must match the
// signal-id assignment in `channel_to_signal_id` below, because the per-channel signal id is
// its offset within its QP's block and the provisioned budget has to cover the largest offset.
//
// QP assignment is two-level, so there are two regimes:
//   * num_sms <= num_available_qps: each SM owns its own block of QPs
//     (`num_qps_in_sm = avail / num_sms`, plus one for the first `avail % num_sms` SMs) and
//     balances only its own channels across them. The worst SM is one without the remainder
//     bonus, so it hosts `ceil(channels_per_sm / (avail / num_sms))` channels per QP.
//   * num_sms >  num_available_qps: all SMs share all QPs, and the global balanced partition
//     gives `ceil(num_channels / avail)` channels per QP.
//
// The first regime is not `ceil(num_sms * channels_per_sm / avail)`: spare QPs owned by other
// SMs cannot absorb this SM's channels. That form provisions too few signals and the kernel
// then signals ids outside the provisioned range, which fails silently (no counts arrive and
// dispatch times out in the CPU wait with all-zero received counts).
__forceinline__ __device__ __host__ constexpr int channels_per_context(
        int num_sms, int num_available_qps, int num_channels_per_sm) {
    const int avail = num_available_qps > 1 ? num_available_qps : 1;
    const int sms = num_sms > 1 ? num_sms : 1;
    if (sms <= avail) {
        const int num_qps_in_sm = avail / sms;
        return math::constexpr_ceil_div(num_channels_per_sm,
                                        num_qps_in_sm > 1 ? num_qps_in_sm : 1);
    }
    return math::constexpr_ceil_div(sms * num_channels_per_sm, avail);
}

// Per-part signal allocation: pick the largest num_parts (up to kMaxParts) that fits
//   channels_per_context(...) * num_parts <= gin_indexed_signals_cnt
// at the requested channels/SM, then reduce channels_per_sm until the budget holds.
__forceinline__ __device__ __host__ constexpr GinPartAllocation compute_part_allocation(
        const GinResourceConfig& cfg, int num_sms, int num_available_qps, int num_channels_per_sm) {
    const int gin_signals = cfg.gin_indexed_signals_cnt;
    const int channels_per_ctx = channels_per_context(num_sms, num_available_qps, num_channels_per_sm);
    const int budget_parts = gin_signals / (channels_per_ctx > 1 ? channels_per_ctx : 1);
    GinPartAllocation alloc{};
    alloc.num_parts = budget_parts < kMaxParts ? budget_parts : kMaxParts;
    alloc.num_parts = alloc.num_parts > 1 ? alloc.num_parts : 1;
    alloc.num_channels_per_sm = num_channels_per_sm;
    while (alloc.num_channels_per_sm > 1 and
           static_cast<long long>(channels_per_context(num_sms, num_available_qps,
                                                      alloc.num_channels_per_sm)) * alloc.num_parts > gin_signals)
        --alloc.num_channels_per_sm;
#ifndef __CUDA_ARCH__
    EP_HOST_ASSERT(static_cast<long long>(channels_per_context(num_sms, num_available_qps,
                                                               alloc.num_channels_per_sm)) * alloc.num_parts <= gin_signals and
                   "GIN signal budget cannot host even 1 part-signal per channel "
                   "at 1 channel/SM. Reduce --num-sms or num_allocated_qps.");
#endif
    return alloc;
}

// Kernel-side entry points: derive the per-channel part count (and verify the launched
// channel count) as compile-time constants from the provisioned indexed-signal budget.
__forceinline__ __device__ __host__ constexpr int constexpr_num_parts(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_parts;
}

__forceinline__ __device__ __host__ constexpr int constexpr_channels_per_sm(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_channels_per_sm;
}

// Channel to QP mapping shared by `get_qp_mode` and `get_qp_signal_id` below. Unlike the
// default kernels' interleaved mapping (`comm::get_qp_mode`), channels are assigned to QPs in
// balanced contiguous blocks, so an SM's channels stay grouped on one GIN context and the
// remainder is spread one channel per QP instead of leaving a trailing QP idle.
#if defined(__CUDACC__)
#define QP_MAPPING_HD __host__ __device__ __forceinline__
#else
#define QP_MAPPING_HD inline
#endif

// Result of a balanced contiguous partition: which bin an item lands in, its
// 0-based index within that bin, and the total size of that bin.
struct QPSlot {
    int bin;
    int local;
    int bin_size;
};

// Balanced contiguous partition of `n` items across `q` bins: the first `n % q` bins own
// `ceil(n/q)` items, the remaining bins own `floor(n/q)`, and items are assigned in
// contiguous blocks (item 0 goes to bin 0). Any two bins differ in size by at most one, no
// bin is idle when `q <= n`, and the bin index is non-decreasing in `idx`.
//
// Callers guarantee `q >= 1`. `base == 0` (`q > n`) is safe: then `rem == n`, `split == n`,
// and every valid `idx` in [0, n) takes the first branch, so `base` is never a divisor.
QP_MAPPING_HD constexpr QPSlot balanced_partition(int idx, int n, int q) {
    const int base = n / q;
    const int rem = n % q;                    // Number of bins with `base + 1` items
    const int split = rem * (base + 1);       // Items owned by those leading bins
    if (idx < split)
        return QPSlot{idx / (base + 1), idx % (base + 1), base + 1};
    const int d = idx - split;
    return QPSlot{rem + d / base, d % base, base};
}

// QP index for a data/notify channel (the NCCL resource-sharing mode is added by `get_qp_mode`)
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps>
QP_MAPPING_HD constexpr int channel_to_qp(int sm_idx, int channel_in_sm_idx,
                                          bool is_notify_warp = false) {
    static_assert(kNumQPs >= 1, "kNumQPs must be >= 1, otherwise balanced_partition() divides by zero");
    // Only one QP
    if constexpr (kNumQPs == 1)
        return 0;

    // The notify warp always uses 1 SM and 1 QP
    if (is_notify_warp)
        return 0;

    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // More QPs than SMs: an SM owns `num_qps_in_sm` QPs, strided across SMs at
        // sm_idx + offset*kNumSMs. Within the SM, balance the SM's channels across
        // its local QPs (contiguous blocks, remainder spread one-per-QP).
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));
        const int local_qp_idx = balanced_partition(channel_in_sm_idx, kNumChannelsPerSM, num_qps_in_sm).bin;
        return kQPStartIdx + sm_idx + local_qp_idx * kNumSMs;
    } else {
        // Fewer QPs than SMs: all SMs share all QPs. Balance the global channels
        // across the QPs (contiguous blocks, remainder spread one-per-QP), so an
        // SM's channels stay grouped and no QP is left idle.
        constexpr int kNumChannels = kNumSMs * kNumChannelsPerSM;
        const int global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return kQPStartIdx + balanced_partition(global_channel_idx, kNumChannels, kNumAvailableQPs).bin;
    }
}

// Per-channel signal id within its QP (0-based index among the channels sharing
// that QP). Uses the same balanced partition as `channel_to_qp`, so `(qp, signal_id)` is
// unique per channel and `signal_id < ceil(channels / qps)`, within the tuner's signal budget.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps>
QP_MAPPING_HD constexpr int channel_to_signal_id(int sm_idx, int channel_in_sm_idx) {
    static_assert(kNumQPs >= 1, "kNumQPs must be >= 1, otherwise balanced_partition() divides by zero");
    if constexpr (kNumQPs == 1)
        return sm_idx * kNumChannelsPerSM + channel_in_sm_idx;

    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));
        return balanced_partition(channel_in_sm_idx, kNumChannelsPerSM, num_qps_in_sm).local;
    } else {
        constexpr int kNumChannels = kNumSMs * kNumChannelsPerSM;
        const int global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return balanced_partition(global_channel_idx, kNumChannels, kNumAvailableQPs).local;
    }
}

#if defined(__CUDACC__)
// `comm::get_qp_mode` with the balanced mapping above: the QP index comes from
// `channel_to_qp`, the sharing mode is CTA when each SM owns its QPs and GPU-wide otherwise
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp = false) {
    constexpr auto kSharingCTA = NCCL_GIN_RESOURCE_SHARING_CTA;
    constexpr auto kSharingGrid = kNumSMs == 1 ? NCCL_GIN_RESOURCE_SHARING_CTA : NCCL_GIN_RESOURCE_SHARING_GPU;
    if constexpr (kNumQPs == 1)
        return {0, kSharingGrid};
    if (is_notify_warp)
        return {0, kSharingCTA};

    constexpr int kNumAvailableQPs = kNumQPs - static_cast<int>(kWithNotifyWarps);
    const int qp_idx = channel_to_qp<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
        sm_idx, channel_in_sm_idx, is_notify_warp);
    if constexpr (kNumSMs <= kNumAvailableQPs)
        return {qp_idx, kSharingCTA};
    else
        return {qp_idx, kSharingGrid};
}

// Unique per-channel signal id within the channel's QP:
//   kNumQPs == 1                : every channel on the single QP, id = global channel index
//   kNumSMs <= kNumAvailableQPs : offset within the SM's balanced local-QP block
//   kNumSMs  > kNumAvailableQPs : offset within the global balanced QP block
// In all cases id < ceil(channels / qps), so `(qp, id)` is unique per channel and the
// tuner's per-QP signal budget (sized for ceil(channels / qps)) is never exceeded.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_qp_signal_id(const int& sm_idx, const int& channel_in_sm_idx) {
    return channel_to_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(sm_idx, channel_in_sm_idx);
}

// Per-part indexed-signal id: kNumParts contiguous ids under the channel's base id, one
// per token part and shared by all sources. The tuner and the channel cap guarantee
// ceil(num_channels / qps) * kNumParts <= gin_indexed_signals_cnt.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, int kNumParts, bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_per_part_signal_id(
    const int& sm_idx, const int& channel_in_sm_idx, const int& part_idx) {
    return get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
               sm_idx, channel_in_sm_idx) * kNumParts + part_idx;
}
#endif

#undef QP_MAPPING_HD

}  // namespace deep_ep::ep::unordered
