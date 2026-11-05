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

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/gin_resource_alloc.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>


namespace deep_ep::elastic {

static constexpr int64_t kScaleoutHeaderWrittenBit = static_cast<int64_t>(1) << 63;

__device__ __forceinline__ int64_t pack_scaleout_header(const int& count, const bool& more) {
    return kScaleoutHeaderWrittenBit
         | (static_cast<int64_t>(count & 0x3fffffff) << 1)
         | (more ? static_cast<int64_t>(1) : static_cast<int64_t>(0));
}

__device__ __forceinline__ bool unpack_scaleout_header(const int64_t& header, int& count, bool& more) {
    count = static_cast<int>((header >> 1) & 0x3fffffff);
    more = (header & 0x1) != 0;
    return header < 0;
}

#ifndef EP_NUM_SUB_PARTS
#define EP_NUM_SUB_PARTS 2
#endif

#ifndef EP_MIN_SUB_TOKENS
#define EP_MIN_SUB_TOKENS 1
#endif

static constexpr int kNumSubPartsDefault = (EP_NUM_SUB_PARTS) > 1 ? (EP_NUM_SUB_PARTS) : 1;
static constexpr int kMinSubTokensDefault = (EP_MIN_SUB_TOKENS) > 1 ? (EP_MIN_SUB_TOKENS) : 1;

#ifndef EP_SM100_MIN_SUB_TOKENS
#define EP_SM100_MIN_SUB_TOKENS 15
#endif

// Experiment knob (JD 23.14 E03): shaped first and last parts (rising and halving sub-puts) on SM100 too
#ifndef EP_SM100_SHAPE_PARTS
#define EP_SM100_SHAPE_PARTS 0
#endif

// Experiment knob (JD 23.14 E03): number of rising sub-puts in the first part, 0 = as many as the last part
#ifndef EP_FIRST_SUB_PARTS
#define EP_FIRST_SUB_PARTS 0
#endif

// Experiment knob (JD 23.14 E03): 0 keeps the first part in uniform sub-puts while the last part is shaped
#ifndef EP_SHAPE_FIRST_PART
#define EP_SHAPE_FIRST_PART 1
#endif

#ifndef EP_MIN_TOKENS_PER_PART
#define EP_MIN_TOKENS_PER_PART 15
#endif

static constexpr int kMinTokensPerPart = (EP_MIN_TOKENS_PER_PART) > 1 ? (EP_MIN_TOKENS_PER_PART) : 1;

template <int kNumSubParts, int kMinSubTokens = kMinSubTokensDefault>
__device__ __host__ __forceinline__ int num_sub_parts_at(const int& part_tokens) {
    if constexpr (kNumSubParts <= 1) {
        return 1;
    } else {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        constexpr int kArchMinSubTokens = (EP_SM100_MIN_SUB_TOKENS) > 1 ? (EP_SM100_MIN_SUB_TOKENS) : 1;
#else
        constexpr int kArchMinSubTokens = 1;
#endif
        constexpr int kEffMinSubTokens = kMinSubTokens > kArchMinSubTokens ? kMinSubTokens : kArchMinSubTokens;
        int n = part_tokens < kNumSubParts ? part_tokens : kNumSubParts;
        if constexpr (kEffMinSubTokens > 1) {
            const int by_floor = part_tokens / kEffMinSubTokens;
            n = n < by_floor ? n : by_floor;
        }
        return n > 1 ? n : 1;
    }
}

// `kHalving`: every sub-part keeps one token and the remaining tokens split 2^(n-1) : ... : 2 : 1,
// so the sizes halve from one sub-part to the next. Otherwise the split is uniform.
template <int kNumSubParts, bool kHalving = false>
__device__ __host__ __forceinline__ int sub_part_size_at(const int& part_tokens, const int& sub) {
    if constexpr (kNumSubParts <= 1) {
        return part_tokens;
    } else {
        const int num_subs = num_sub_parts_at<kNumSubParts>(part_tokens);
        if constexpr (kHalving) {
            const int spare = part_tokens - num_subs;
            const int total = (1 << num_subs) - 1;
            int assigned = 0, mine = 0;
            for (int s = 0; s < num_subs; ++ s) {
                const int v = spare * (1 << (num_subs - 1 - s)) / total;
                assigned += v;
                mine = s == sub ? v : mine;
            }
            return 1 + mine + (sub == 0 ? spare - assigned : 0);
        }
        const int base = part_tokens / num_subs;
        const int rem = part_tokens - base * num_subs;
        return base + (sub < rem ? 1 : 0);
    }
}

template <int kNumSubParts, bool kHalving = false>
__device__ __host__ __forceinline__ int sub_part_offset_at(const int& part_tokens, const int& sub) {
    if constexpr (kNumSubParts <= 1) {
        return 0;
    } else {
        int off = 0;
        for (int s = 0; s < sub; ++ s)
            off += sub_part_size_at<kNumSubParts, kHalving>(part_tokens, s);
        return off;
    }
}

template <int kNumParts, int kBatchSize, int kNumMaxTokensPerChannel>
__device__ __host__ __forceinline__ constexpr int part_size_at(const int& part) {
    constexpr int kFirst = kBatchSize / 3 > 4 ? kBatchSize / 3 : 4;
    constexpr int kMidParts = kNumParts - 2;
    constexpr int kMidTotal = kNumMaxTokensPerChannel - kFirst - kBatchSize;
    if constexpr (kNumParts < 3 or kMidTotal < kMidParts) {
        return kBatchSize;
    } else {
        if (part == 0)
            return kFirst;
        if (part == kNumParts - 1)
            return kBatchSize;
        const int mid_idx = part - 1;
        return kMidTotal / kMidParts + (mid_idx < (kMidTotal % kMidParts) ? 1 : 0);
    }
}

template <int kNumParts, int kBatchSize, int kNumMaxTokensPerChannel>
__device__ __host__ __forceinline__ constexpr int part_offset_at(const int& part) {
    int off = 0;
    for (int p = 0; p < part; ++ p)
        off += part_size_at<kNumParts, kBatchSize, kNumMaxTokensPerChannel>(p);
    return off;
}

// QP 0 carries the rail barrier's indexed signals. The data channels stay above QP 0 in
// cached mode too (no notify warps), matching the host tuner and the combine kernel.
static constexpr bool kDataChannelsAboveQP0 = true;

template <bool kDoCPUSync,
          bool kReuseSlotIndices,
          bool kAllowMultipleReduction,
          bool kDoExpand,
          bool kDoubleBufferForward,
          int kNumSMs,
          int kNumNotifyWarps, int kNumScaleoutWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumGinSignals,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumChannelsPerSM = kNumScaleoutWarps,
          int kNumChannels = kNumScaleoutWarps * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kNumBudgetParts = gin_alloc::constexpr_num_parts(
              kNumGinSignals, kNumSMs, kNumQPs, kDataChannelsAboveQP0, kNumScaleoutWarps),
          int kNumGeomParts = kMinTokensPerPart <= 1 ? kNumBudgetParts
                            : ((kNumMaxTokensPerChannel / kMinTokensPerPart > 1)
                               ? kNumMaxTokensPerChannel / kMinTokensPerPart : 1),
          int kNumParts = kNumBudgetParts < kNumGeomParts ? kNumBudgetParts : kNumGeomParts,
          int kPartSize = math::constexpr_ceil_div(kNumMaxTokensPerChannel, kNumParts),
          int kBatchSize = kPartSize,
          int kNumSubParts = kNumSubPartsDefault < kBatchSize ? kNumSubPartsDefault : kBatchSize,
          int kNumSlotsPerForwardChunk = 16,
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,
          int kNumForwardThreads = kNumForwardWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_unordered_dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    int* token_metadata_at_forward,
    int* token_map_at_dispatch,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    // TODO(NCCL): so many params, plans to optimize?
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    void* buffer,
    void* workspace, void* mapped_host_workspace,
    const int scaleout_rank_idx, const int scaleup_rank_idx) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp size");
    EP_STATIC_ASSERT(kNumParts <= layout::WorkspaceLayout::kNumMaxParts,
                     "kNumParts exceeds the per-part header workspace capacity");
    EP_STATIC_ASSERT(kNumParts >= 1, "Invalid part count");
    EP_STATIC_ASSERT(kNumSubParts >= 1, "Invalid sub-part count");
    EP_STATIC_ASSERT(kNumSubParts <= kBatchSize,
                     "More sub-parts than tokens in a part: every sub-part must own >= 1 token "
                     "so its header has its own slot");
    EP_STATIC_ASSERT(gin_alloc::constexpr_channels_per_sm(
                         kNumGinSignals, kNumSMs, kNumQPs, kDataChannelsAboveQP0, kNumScaleoutWarps)
                         == kNumScaleoutWarps,
                     "GIN signal budget cannot host the launched channel count");

    // Utils
    // NOTES: a warp is a channel (different channels may share QPs)
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // NCCL Gin handle
    // Each warp is a channel
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, kDataChannelsAboveQP0>(
        sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto scaleout_token_layout = layout::TokenLayout(
        kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true, nullptr, /*with_scaleout_hdr=*/true);
    constexpr int kNumFwdBuffers = kDoubleBufferForward ? kNumDispatchFwdBuffers : 1;

    // Per channel, after the TMA pool: the routing of one forward chunk (one token per lane: rank, slot and
    // linked-list index per (token, top-k), the token's prefix per scale-up rank, the chunk's slot base and
    // linked-list counter per scale-up rank) and the mailbox between the forward warp and its helper (posted
    // and done chunk counts, the forwarder's finish flag, the helper's ready flag, the chunk's token count
    // and the forwarder's share, then the helper's source-token, metadata and slot-index pointers)
    constexpr int kNumRouteTokens = kNumSlotsPerForwardChunk;
    EP_STATIC_ASSERT(kNumRouteTokens <= 32, "One lane routes one token of a forward chunk");
    constexpr int kNumRouteInts = math::constexpr_align(
        kNumRouteTokens * (kNumTopk * 3 + kNumScaleupRanks) + kNumScaleupRanks * 2, 4);
    constexpr int kNumMailboxInts = 16;
    enum { kMbPosted = 0, kMbDone, kMbFwdDone, kMbReady, kMbNumTokens, kMbFwdShare, kMbPtrs = 8 };
    constexpr int kNumSmemTokenBytes =
        math::constexpr_align(kNumHiddenBytes, ptx::kNumTMAAlignBytes) +
        math::constexpr_align(kNumSFPacks * static_cast<int>(sizeof(sf_pack_t)), ptx::kNumTMAAlignBytes) +
        math::constexpr_align(kNumTopk * 8 + (1 + kNumTopk) * 4, ptx::kNumTMAAlignBytes) +
        math::constexpr_align(static_cast<int>(sizeof(ptx::mbarrier)), ptx::kNumTMAAlignBytes);
    constexpr int kNumMaxSmemBytes = 232448;
    constexpr int kNumBasePoolBuffers = kNumScaleoutWarps + kNumFwdBuffers * kNumForwardWarps;
    constexpr int kNumRouteSmemBytes = kNumForwardWarps * (kNumRouteInts + kNumMailboxInts) * static_cast<int>(sizeof(int));
    // The host fills shared memory with channels, so the chunk routing (and the forward helper that needs
    // it) is used where its tables fit, and the per-token routing otherwise. With a double-buffered forward
    // the helper also gets a second buffer where that fits.
    constexpr bool kChunkRouting =
        kNumScaleoutRanks > 1 and
        kNumSmemBytesForNotify + kNumBasePoolBuffers * kNumSmemTokenBytes + kNumRouteSmemBytes <= kNumMaxSmemBytes;
    constexpr bool kHelperDoubleBuffer =
        kChunkRouting and kDoubleBufferForward and
        kNumSmemBytesForNotify + (kNumBasePoolBuffers + kNumScaleoutWarps) * kNumSmemTokenBytes + kNumRouteSmemBytes <=
            kNumMaxSmemBytes;
    constexpr int kNumPoolBuffers = kNumBasePoolBuffers + (kHelperDoubleBuffer ? kNumScaleoutWarps : 0);

    const auto tma_pool = layout::BufferLayout<true>(
        token_layout, kNumPoolBuffers, 1,
        math::advance_ptr<int>(smem, kNumSmemBytesForNotify));
    const bool is_forward_warp = warp_idx >= kNumNotifyWarps + kNumScaleoutWarps;
    const int fwd_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);
    const int tma_pool_base = is_forward_warp
        ? kNumScaleoutWarps + kNumFwdBuffers * fwd_warp_idx
        : (warp_idx - kNumNotifyWarps);
    const auto tma_buffer = tma_pool.get_rank_buffer(tma_pool_base).get_token_buffer(0);
    const auto tma_buffer_alt = tma_pool.get_rank_buffer(
        kDoubleBufferForward and is_forward_warp ? tma_pool_base + 1 :
        kHelperDoubleBuffer and not is_forward_warp ? kNumBasePoolBuffers + (warp_idx - kNumNotifyWarps) :
        tma_pool_base).get_token_buffer(0);

    // The sender warp of a channel shares its forward warp's tables (it helps once its tokens are staged)
    const int chan_warp_idx = is_forward_warp ? fwd_warp_idx : (warp_idx >= kNumNotifyWarps ? warp_idx - kNumNotifyWarps : 0);
    int* const route_rank = static_cast<int*>(tma_pool.get_buffer_end_ptr()) +
        chan_warp_idx * (kNumRouteInts + kNumMailboxInts);
    int* const route_slot = route_rank + kNumRouteTokens * kNumTopk;
    int* const route_ll = route_slot + kNumRouteTokens * kNumTopk;
    int* const route_prefix = route_ll + kNumRouteTokens * kNumTopk;
    int* const route_base = route_prefix + kNumRouteTokens * kNumScaleupRanks;
    int* const route_count = route_base + kNumScaleupRanks;
    int* const mailbox = route_rank + kNumRouteInts;

    constexpr int kNumSlotsPerChannel = kNumParts * kBatchSize;
    const auto part_size = [&](const int& part) {
        return part_size_at<kNumParts, kBatchSize, kNumMaxTokensPerChannel>(part);
    };
    const auto part_first_slot = [&](const int& part) {
        return part_offset_at<kNumParts, kBatchSize, kNumMaxTokensPerChannel>(part);
    };
    // The last part lands at the end of the transfer: it splits into twice as many sub-puts with
    // halving sizes, so few tokens are left to forward after the last landing. SM100 keeps the
    // uniform split, its sub-puts have a token floor.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr bool kShapeLastPart = (EP_SM100_SHAPE_PARTS) != 0;
#else
    constexpr bool kShapeLastPart = true;
#endif
    constexpr int kNumLastSubParts = kShapeLastPart ? 2 * kNumSubParts : kNumSubParts;
    constexpr int kNumFirstSubParts = (EP_FIRST_SUB_PARTS) > 0 ? (EP_FIRST_SUB_PARTS) : kNumLastSubParts;
    EP_STATIC_ASSERT(kNumFirstSubParts <= kNumLastSubParts, "The first part has at most the last part's sub-puts");
    constexpr int kNumMaxSubParts = kNumLastSubParts > kNumSubParts ? kNumLastSubParts : kNumSubParts;
    // Without a small first part (two parts per channel) the first part is cut the other way round,
    // in sub-puts of rising size, so the first put leaves after a few staged tokens.
    constexpr bool kShapeFirstPart = kShapeLastPart and (EP_SHAPE_FIRST_PART) != 0 and kNumParts >= 2 and
        part_size_at<kNumParts, kBatchSize, kNumMaxTokensPerChannel>(0) == kBatchSize;
    const auto num_subs_of = [&](const int& part_idx) {
        if (kShapeFirstPart and part_idx == 0)
            return num_sub_parts_at<kNumFirstSubParts>(part_size(0));
        return part_idx == kNumParts - 1
            ? num_sub_parts_at<kNumLastSubParts>(part_size(part_idx))
            : num_sub_parts_at<kNumSubParts>(part_size(part_idx));
    };
    const auto sub_size_of = [&](const int& part_idx, const int& sub_idx) {
        if (kShapeFirstPart and part_idx == 0)
            return sub_part_size_at<kNumFirstSubParts, true>(
                part_size(0), num_sub_parts_at<kNumFirstSubParts>(part_size(0)) - 1 - sub_idx);
        return part_idx == kNumParts - 1 ? sub_part_size_at<kNumLastSubParts, kShapeLastPart>(part_size(part_idx), sub_idx)
                                         : sub_part_size_at<kNumSubParts>(part_size(part_idx), sub_idx);
    };
    // First slot of sub-part `sub_idx` within part `part_idx` (slots inside a sub-part are
    // filled contiguously from here; its in-band header lives in this slot's header word).
    const auto sub_slot = [&](const int& part_idx, const int& sub_idx) {
        if (kShapeFirstPart and part_idx == 0) {
            int off = 0;
            for (int s = 0; s < sub_idx; ++ s)
                off += sub_size_of(0, s);
            return off;
        }
        return part_first_slot(part_idx) +
            (part_idx == kNumParts - 1 ? sub_part_offset_at<kNumLastSubParts, kShapeLastPart>(part_size(part_idx), sub_idx)
                                       : sub_part_offset_at<kNumSubParts>(part_size(part_idx), sub_idx));
    };
    // Per-(channel, part) counting-signal id, shared by the sender and forward sides of the
    // channel this warp serves (both sides reduce to the same channel warp index).
    const auto part_signal_id = [&](const int& part_idx) {
        return static_cast<ncclGinSignal_t>(
            comm::get_per_part_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM,
                                         kNumParts, kDataChannelsAboveQP0>(
                sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, part_idx));
    };

    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        scaleout_token_layout, kNumScaleoutRanks, kNumChannels * kNumSlotsPerChannel, scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        scaleout_token_layout, kNumScaleoutRanks, kNumChannels * kNumSlotsPerChannel, scaleout_send_buffer.get_buffer_end_ptr());

    // Split entry: the first forward warp of SM 0 clears the in-band headers of the remote sources, runs
    // the NVLink and rail handshakes and opens the entry flag. The header words live in `buffer`, which the
    // combine and other kernels also write between two dispatches, so they hold arbitrary data until this
    // clear. No remote sender writes them before it passes the rail barrier, and the release fence orders
    // the zeros before this GPU's barrier signal. Warps stage into their own buffers right away. Every write
    // into a peer's buffers and every header poll of a forwarder waits for the flag. The flag goes back to 0
    // at kernel exit, so one constant marks an open entry, also when a CUDA graph replays the kernel.
    constexpr int kEntryOpen = 1;
    const auto entry_flag_ptr = workspace_layout.get_dispatch_entry_flag_ptr();
    const auto wait_entry = [&]() {
        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            if (ptx::ld_acquire_sys<int>(entry_flag_ptr) == kEntryOpen)
                return true;
            if (is_last_check and lane_idx == 0) {
                printf("DeepEP hybrid dispatch (entry flag) timeout, scale-out: %d, scale-up: %d, "
                       "sm: %d, warp: %d, flag: %d\n",
                       scaleout_rank_idx, scaleup_rank_idx, sm_idx, warp_idx,
                       ptx::ld_acquire_sys<int>(entry_flag_ptr));
            }
            return false;
        });
    };
    if (sm_idx == 0 and warp_idx == kNumNotifyWarps + kNumScaleoutWarps) {
        if constexpr (kNumScaleoutRanks > 1) {
            for (int i = lane_idx; i < kNumChannels * kNumScaleoutRanks; i += 32) {
                const int src_rank = i % kNumScaleoutRanks;
                if (src_rank == scaleout_rank_idx)
                    continue;
                const auto src_buffer = scaleout_recv_buffer
                    .get_channel_buffer<kNumSlotsPerChannel>(i / kNumScaleoutRanks)
                    .get_rank_buffer(src_rank);
                #pragma unroll
                for (int k = 0; k < kNumParts; ++ k) {
                    for (int s = 0; s < num_subs_of(k); ++ s)
                        *src_buffer.get_token_buffer(sub_slot(k, s)).get_hdr_ptr() = 0;
                }
            }
            ptx::fence_release_sys();
        }
        __syncwarp();
        if constexpr (kNumScaleupRanks > 1) {
            comm::nvlink_barrier_warp<kNumScaleupRanks, kNumScaleupRanksPerLane, kNumTimeoutCycles,
                                      comm::kHybridDispatchTag0>(
                gin, workspace_layout, scaleup_rank_idx, lane_idx);
        }
        if constexpr (kNumScaleoutRanks > 1) {
            comm::gin_barrier_wo_local_sync<kNumScaleoutRanks, 1, 32, kNumQPs, kNumTimeoutCycles,
                                            ncclTeamTagRail, comm::kHybridDispatchTag0, false>(
                gin.nccl_dev_comm, scaleout_rank_idx, scaleup_rank_idx, 0, lane_idx);
        }
        __syncwarp();
        if (lane_idx == 0)
            ptx::st_release_sys(entry_flag_ptr, kEntryOpen);
        __syncwarp();
    }

    // Init TMA for scale-out and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    ptx::arrival_phase phase_alt = 0;
    const auto mbarrier_ptr_alt = tma_buffer_alt.get_mbarrier_ptr();
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync()) {
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        if ((kDoubleBufferForward and is_forward_warp) or (kHelperDoubleBuffer and not is_forward_warp))
            ptx::mbarrier_init_with_fence(mbarrier_ptr_alt, 1);
    }
    if constexpr (kChunkRouting) {
        if (warp_idx >= kNumNotifyWarps and not is_forward_warp and lane_idx < kNumMailboxInts)
            mailbox[lane_idx] = 0;
        __syncthreads();
    } else {
        __syncwarp();
    }

    // Forward one token that sits in `fwd_token` (scale-out layout) with its routing at row `route_row`:
    // widen the top-k indices, move the weights and the source index to their scale-up offsets, add the
    // linked-list indices, store it into its scale-up ranks and record its forward metadata. Run by the
    // forward warp and by its helper on the same channel.
    const int fwd_channel_idx = sm_idx * kNumChannelsPerSM + chan_warp_idx;
    const auto fwd_scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);
    const auto forward_token = [&](const layout::TokenLayout& fwd_token, const int& route_row,
                                   int* metadata_ptr, int* dst_slot_idx_ptr, const bool& is_last)
                                   __attribute__((always_inline)) {
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        const auto received = layout::TokenLayout(
            kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true, fwd_token.get_base_ptr(), true);
        const int route_idx = route_row * kNumTopk + lane_idx;
        int dst_expert_idx = -1;
        float topk_weight = 0;
        int dst_scaleup_rank_idx = -1, dst_slot_idx = -1, linked_list_idx = -1;
        if (lane_idx < kNumTopk) {
            dst_expert_idx = received.get_scaleout_topk_idx_ptr()[lane_idx];
            topk_weight = received.get_topk_weights_ptr()[lane_idx];
            dst_scaleup_rank_idx = route_rank[route_idx];
            dst_slot_idx = route_slot[route_idx];
            linked_list_idx = route_ll[route_idx];
        }
        const int src_token_global_idx = *received.get_src_token_global_idx_ptr();
        __syncwarp();
        if (lane_idx < kNumTopk) {
            fwd_token.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
            fwd_token.get_topk_weights_ptr()[lane_idx] = topk_weight;
            if constexpr (not kReuseSlotIndices) {
                constexpr int kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1;
                fwd_token.get_linked_list_idx_ptr()[lane_idx] =
                    fwd_channel_idx * (kNumTokensInLinkedList * kNumScaleupRanks) +
                    linked_list_idx * kNumScaleupRanks + scaleup_rank_idx;
            }
        }
        if (lane_idx == 0)
            *fwd_token.get_src_token_global_idx_ptr() = src_token_global_idx;
        ptx::tma_store_fence();
        __syncwarp();

        if (dst_slot_idx >= 0) {
            const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                fwd_scaleup_buffer.get_token_buffer(dst_slot_idx).get_base_ptr(), dst_scaleup_rank_idx);
            ptx::tma_store_1d(dst_ptr, fwd_token.get_base_ptr(), fwd_token.get_num_bytes<false>());
            ptx::tma_store_commit();
        }
        __syncwarp();

        if constexpr (not kReuseSlotIndices) {
            if (ptx::elect_one_sync()) {
                metadata_ptr[0] = src_token_global_idx;
                metadata_ptr[1] = is_last;
            }
            if (lane_idx < kNumTopk) {
                metadata_ptr[2 + lane_idx] = dst_scaleup_rank_idx;
                metadata_ptr[2 + kNumTopk + lane_idx] = dst_slot_idx;
                dst_slot_idx_ptr[lane_idx] = dst_slot_idx;
            }
        }
        __syncwarp();
    };

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // Assign shared memory
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Do full-grid reduction
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0
        if (sm_idx == 0) {
            // Reduce all SM's count
            // Wait all SMs' arrival
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        // Encode and write into the send buffer
                        workspace_layout.get_scaleout_rank_expert_count_ptr<true>()[i] =
                            math::encode_decode_positive<int>(status & 0xffffffffll);

                        // Clean for the next usage
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("DeepEP hybrid notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            wait_entry();

            // Issue scaleout writes to peers
            EP_STATIC_ASSERT(kReuseSlotIndices or kNumScaleoutRanks <= kNumNotifyThreads,
                             "kNumScaleoutRanks must be less than kNumNotifyThreads");
            if (thread_idx < kNumScaleoutRanks) {
                const auto dst_scaleout_rank_idx = thread_idx;
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,
                    ncclGinOptFlagsAggregateRequests);
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx);
            }
            __syncwarp();

            // Util functions to get metadata from scale-out peers
            // NOTES: this is correct as RDMA operations has a minimum write granularity of 1024 bytes (a whole integer write is atomic)
            const auto recv_and_reduce = [=](const auto& get_ptr_func, const bool& is_expert_reduction = false) -> int {
                int count = 0;
                #pragma unroll
                for (int j = 0; j < kNumScaleoutRanks; ++ j) {
                    const auto ptr = get_ptr_func(j);
                    int decoded;
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){
                        decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));
                        if (math::is_decoded_positive_ready(decoded))
                            return true;

                        if (is_last_check) {
                            printf("DeepEP hybrid notify (scale-out %s reduction) timeout, "
                                   "scale-out: %d, scale-up: %d, "
                                   "thread: %d, wait scale-out: %d, decoded: %d\n",
                                   is_expert_reduction ? "expert" : "rank",
                                   scaleout_rank_idx, scaleup_rank_idx, thread_idx, j,
                                   decoded);
                        }
                        return false;
                    });

                    // Add and clean for next usages
                    count += decoded, *ptr = 0;
                }
                return count;
            };

            // Write into all scale-up peers' rank-level counters
            #pragma unroll
            for (int i = thread_idx; i < kNumScaleupRanks; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
                });

                // Write into the remote scale-up peer
                const int64_t counter = (static_cast<int64_t>(kNumScaleupRanks) << 32ll) | count;
                gin.put_value<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
                    counter, i);
            }
            __syncwarp();

            // Atomic add into all scale-up peers' expert-level counters
            #pragma unroll
            for (int i = thread_idx; i < kNumExpertsPerScaleout; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                }, true);

                // Write into the remote scale-up peer
                const int64_t counter = (1ll << 32ll) | count;
                const auto dst_scaleup_rank_idx = i / kNumExpertsPerRank;
                const auto expert_idx_in_dst_rank = i % kNumExpertsPerRank;
                gin.red_add_rel<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
                    counter, dst_scaleup_rank_idx);
            }
            // There are shared memory reads above, a barrier is necessary
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // NOTES: from now on, the `rank` and `expert`s size change into the local size
            expert_count = rank_expert_count + kNumScaleupRanks;

            // Wait local counters to be ready
            // NOTES: here we only care the prefix sum by scale-up peers (used for later epilogue), not all ranks
            EP_STATIC_ASSERT(kNumNotifyWarps == 0 or kNumScaleupRanks + kNumExpertsPerRank <= kNumNotifyWarps * 32,
                             "Insufficient notify threads");
            comm::timeout_while<kNumTimeoutCycles>(thread_idx < kNumScaleupRanks + kNumExpertsPerRank,
                [&](const bool& is_last_check) {
                const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + thread_idx);
                if ((status >> 32ull) == kNumScaleupRanks) {
                    // Clean GPU workspace and write into host workspace
                    const auto count = static_cast<int>(status & 0xffffffffll);
                    const auto aligned_count = math::align<int>(
                        count, thread_idx < kNumScaleupRanks ? 1 : kExpertAlignment);

                    workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] = 0;
                    if constexpr (kDoCPUSync) {
                        host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] =
                            math::encode_decode_positive(aligned_count);
                    }

                    // Update statistics counters
                    if (cumulative_local_expert_recv_stats != nullptr and thread_idx >= kNumScaleupRanks)
                        atomicAdd(cumulative_local_expert_recv_stats + (thread_idx - kNumScaleupRanks), count);

                    // Write unaligned count before aligning
                    if (num_unaligned_recv_tokens_per_expert != nullptr and thread_idx >= kNumScaleupRanks)
                        num_unaligned_recv_tokens_per_expert[thread_idx - kNumScaleupRanks] = count;

                    // Save for later prefix sum calculation
                    rank_expert_count[thread_idx] = aligned_count;
                    return true;
                }

                if (is_last_check) {
                    printf("DeepEP hybrid notify (scale-up reduction) timeout,"
                           "scale-out: %d/%d, scale-up: %d/%d, "
                           "thread: %d, status: %d | %d, expected: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                           static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumScaleupRanks);
                }
                return false;
            });
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Do prefix sum by the warps of the first SM
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // Store into global memory
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    psum = ptx::exchange(sum, 31);
                }
            };
            if (warp_idx == 0) {
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumScaleupRanks, 0);
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }

            comm::flush_with_timeout<kNumTimeoutCycles>(gin.gin, ncclCoopWarp());
        }
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        const int scaleout_warp_idx = warp_idx - kNumNotifyWarps;
        const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;
        const auto stage_part_header = [&](layout::BufferLayout<false>& send_ch,
                                           const int& slot, const int& count, const bool& more) {
            auto* hdr = send_ch.get_token_buffer(slot).get_hdr_ptr();
            *hdr = pack_scaleout_header(count, more);
            ptx::fence_acq_rel_sys();
        };
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumSlotsPerChannel>(channel_idx);

        // Channel metadata maintenance
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid number of scale-out ranks");
        int stored_scaleout_tail = 0, stored_old_scaleout_tail = 0;
        int stored_combine_slot_tail = 0;
        int coalesce_flushed = 0;
        int stored_flushed_parts = 0;
        int stored_flushed_subs = 0;
        bool entry_open = false;
        const bool is_remote_dst = lane_idx < kNumScaleoutRanks and lane_idx != scaleout_rank_idx;
        const auto cur_sub_size = [&]() {
            const int p = stored_flushed_parts < kNumParts ? stored_flushed_parts : kNumParts - 1;
            return sub_size_of(p, stored_flushed_subs);
        };
        const auto is_last_sub_put = [&]() {
            if (stored_flushed_parts != kNumParts - 1)
                return false;
            return stored_flushed_subs + 1 >= num_subs_of(kNumParts - 1);
        };
        // Kept inline: as a subroutine (NVCC 13.1 out-lines it at 4 call sites) its by-reference
        // captures move into local memory and the stack frame grows from 432 B to 1280 B
        const auto flush_part = [&](const int& up_to, const bool& is_final) __attribute__((always_inline)) {
            if (lane_idx >= kNumScaleoutRanks or lane_idx == scaleout_rank_idx)
                return;
            const int part_idx = stored_flushed_parts;
            const int sub_idx = stored_flushed_subs;
            const int count = up_to - coalesce_flushed;
            const int slot = sub_slot(part_idx, sub_idx);
            auto send_ch = scaleout_send_buffer.get_rank_buffer(lane_idx)
                               .get_channel_buffer<kNumSlotsPerChannel>(channel_idx);
            stage_part_header(send_ch, slot, count, /*more=*/not is_final);
            const int num_put_tokens = count > 0 ? count : 1;
            gin.put<ncclTeamTagRail>(
                scaleout_recv_buffer.get_token_buffer(slot).get_base_ptr(),
                send_ch.get_token_buffer(slot).get_base_ptr(),
                scaleout_token_layout.get_num_bytes<false>() * num_put_tokens,
                lane_idx, 0,
                ncclGin_SignalAdd{part_signal_id(part_idx), 1ull});
            coalesce_flushed = up_to;
            if (sub_idx + 1 >= num_subs_of(part_idx)) {
                stored_flushed_subs = 0;
                stored_flushed_parts += 1;
            } else {
                stored_flushed_subs = sub_idx + 1;
            }
        };
        const auto update_scaleout_tail = [&](const bool& finish_flag = false) {
            if (lane_idx == scaleout_rank_idx and
                (stored_scaleout_tail >= stored_old_scaleout_tail + kBatchSize or finish_flag)) {
                const auto signaled_tail = math::pack2<int, int64_t>(finish_flag, stored_scaleout_tail);
                const auto ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, scaleout_rank_idx);
                const auto old_signaled_tail = math::pack2<int, int64_t>(0, stored_old_scaleout_tail);

                // NOTES: the "release" scope will be `sys` for the local rank (we may involve NVLink so not `gpu`)
                // For RDMA requests, "release" is ensured by "atomic"
                gin.red_add_rel<ncclTeamTagRail>(ptr, signaled_tail - old_signaled_tail, lane_idx);
                stored_old_scaleout_tail = stored_scaleout_tail;
            }
            __syncwarp();
        };

        // The staging buffer holds the token in the scale-out layout
        const auto staged_token = layout::TokenLayout(
            kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true, tma_buffer.get_base_ptr(), /*with_scaleout_hdr=*/true);

        // Preload next token
        const auto preload_next_token = [&](const int& token_idx) {
            if (token_idx >= num_tokens)
                return;

            // Issue TMA load
            const auto token_i64_idx = static_cast<int64_t>(token_idx);
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue SF `cp.async`
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }
        };

        // Iterate all tokens
        preload_next_token(channel_idx);
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {
            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_scaleout_rank_idx = -1, stored_dst_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerScaleout : -1;
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
                staged_token.get_scaleout_topk_idx_ptr()[lane_idx] = static_cast<int16_t>(dst_expert_idx);
                if (topk_weights != nullptr)
                    staged_token.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            if (ptx::elect_one_sync())
                *staged_token.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
            const auto stored_old_slot_idx = ptx::exchange(
                stored_scaleout_tail, stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);
            if (ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and stored_dst_scaleout_rank_idx >= 0)
                stored_dst_slot_idx = stored_old_slot_idx;

            // Update scale-out tail
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);
            stored_scaleout_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // Build the map that combine will use to gather partials back.
            //
            // Terminology:
            //   T          — token index (per-rank, this rank's PyTorch tensor slot)
            //   k          — top-k slot within a token
            //   G          — global rank id (0 .. ScaleoutRank × ScaleupRank - 1)
            //   scaleout   — G / ScaleupRank, i.e. which node this global rank sits on
            //
            // Setup: T=0, channel=7, topK=4, ScaleoutRank=8, ScaleUpRank=8 (64 global ranks total).
            //   Routing: (k=0 → G=9 → scaleout=1),
            //            (k=1 → G=2 → scaleout=0),
            //            (k=2 → G=5 → scaleout=0),   ← same scaleout as k=1
            //            (k=3 → G=20 → scaleout=2).
            //
            // Each map entry is a `pack(scaleout_rank, slot, channel)` tuple naming the recv slot
            // where combine will place this k's return partial on my `scaleout_recv_buffer`.
            //
            // Reduce mode: k's targeting the same scaleout rank share one recv slot (combine
            // reduces their partials into one before RDMA):
            //   map[0][0..3] = [ pack(1,0,7), pack(0,0,7), pack(0,0,7), pack(2,0,7) ]
            //                                              ^ k=2 shares k=1's slot on scaleout=0
            //
            // Expand mode: each valid k gets its own slot within its scaleout rank's group,
            // packed contiguously:
            //   map[0][0..3] = [ pack(1,0,7), pack(0,0,7), pack(0,1,7), pack(2,0,7) ]
            //                                              ^ k=2 lands at slot=1 next to k=1's slot=0
            //
            // Plain mode (no expand, no multiple reduction): one slot per destination GLOBAL
            // rank. k's routed to the same global rank share their master's slot; k's on
            // different global ranks get their own slots even when those ranks sit on the
            // same scaleout rank. With the routing above (k=1 -> G=2 and k=2 -> G=5 are
            // different global ranks) the map matches the expand one:
            //   map[0][0..3] = [ pack(1,0,7), pack(0,0,7), pack(0,1,7), pack(2,0,7) ]
            // If k=2 instead targeted G=2 (same global rank as k=1), they would share:
            //   map[0][0..3] = [ pack(1,0,7), pack(0,0,7), pack(0,0,7), pack(2,0,7) ]
            //                                              ^ k=2 shares k=1's slot (same G)
            if constexpr (not kReuseSlotIndices) {
                int combine_slot;

                const int combine_slot_base = ptx::exchange(
                    stored_combine_slot_tail,
                    stored_dst_scaleout_rank_idx >= 0 ? stored_dst_scaleout_rank_idx : 0);

                // If reduction enabled, then each token consume signle slot per scaleout rank.
                if constexpr (kAllowMultipleReduction) {
                    combine_slot = combine_slot_base;
                } else if constexpr (kDoExpand) {
                    // Expanded combine returns one partial per valid k, each at its own slot.
                    const uint32_t same_rank_mask = ptx::match(stored_dst_scaleout_rank_idx);
                    const uint32_t below_mask = same_rank_mask & ((1u << lane_idx) - 1u);
                    combine_slot = combine_slot_base + __popc(below_mask);
                } else {
                    // Plain combine returns one partial per destination rank, so k's sharing a
                    // rank share their master's slot. Index by the master lane, which makes the
                    // offset identical for every k of the group.
                    const uint32_t master_mask = ptx::gather(
                        ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0);
                    const uint32_t same_scaleout_mask = ptx::match(stored_dst_scaleout_rank_idx);
                    const int master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));
                    const uint32_t below_mask = (1u << master_lane_idx) - 1u;
                    combine_slot = combine_slot_base + __popc(master_mask & same_scaleout_mask & below_mask);
                }

                if (lane_idx < kNumTopk) {
                    const int packed = stored_dst_scaleout_rank_idx >= 0
                        ? pack_combine_recv_addr(stored_dst_scaleout_rank_idx, combine_slot, channel_idx)
                        : -1;
                    token_map_at_dispatch[token_idx * kNumTopk + lane_idx] = packed;
                }
                __syncwarp();

                if constexpr (kAllowMultipleReduction) {
                    stored_combine_slot_tail += (scaleout_rank_mask >> lane_idx) & 1;
                } else {
                    const uint32_t counted_mask = kDoExpand ?
                        0xffffffffu :
                        ptx::gather(ptx::deduplicate(stored_dst_rank_idx, lane_idx));
                    int my_targeted_count = 0;
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        const int R_k = __shfl_sync(0xffffffff, stored_dst_scaleout_rank_idx, k);
                        const bool counted = (counted_mask >> k) & 1u;
                        my_targeted_count += (R_k == lane_idx and R_k >= 0 and counted) ? 1 : 0;
                    }
                    stored_combine_slot_tail += my_targeted_count;
                }
            }

            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
                auto send_ch = scaleout_send_buffer.get_rank_buffer(stored_dst_scaleout_rank_idx)
                                   .get_channel_buffer<kNumSlotsPerChannel>(channel_idx);
                ptx::tma_store_1d(send_ch.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), scaleout_token_layout.get_num_bytes<false>());
            }
            __syncwarp();

            // Local rank can be bypassed
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx) {
                ptx::tma_store_1d(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), scaleout_token_layout.get_num_bytes<false>());
            }
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            __syncwarp();

            // Preload the next token (overlapping with the scale-out put issues)
            preload_next_token(token_idx + kNumChannels);

            // Puts wait for the entry flag; the sub-parts staged until then go out one by one
            if (not entry_open)
                entry_open = ptx::exchange(static_cast<int>(ptx::ld_acquire_sys<int>(entry_flag_ptr) == kEntryOpen), 0) != 0;
            if (entry_open) {
                while (is_remote_dst and stored_flushed_parts < kNumParts and
                       stored_scaleout_tail >= coalesce_flushed + cur_sub_size())
                    flush_part(coalesce_flushed + cur_sub_size(), /*is_final=*/is_last_sub_put());
            }
            __syncwarp();

            // Issue scale-out tail update
            update_scaleout_tail();
        }

        wait_entry();
        while (is_remote_dst and stored_flushed_parts < kNumParts and
               stored_scaleout_tail >= coalesce_flushed + cur_sub_size())
            flush_part(coalesce_flushed + cur_sub_size(), /*is_final=*/is_last_sub_put());
        if (stored_scaleout_tail > coalesce_flushed) {
            flush_part(stored_scaleout_tail, /*is_final=*/true);
        } else if (stored_flushed_parts < kNumParts) {
            flush_part(coalesce_flushed, /*is_final=*/true);
        }
        __syncwarp();

        // Flush unflushed tails
        update_scaleout_tail(true);

        // Help the forward warp of this channel: forward the second half of every chunk it posts, from
        // the staging buffer, which is idle once the last token is staged (its stores were waited for)
        if constexpr (kChunkRouting) {
            if (ptx::elect_one_sync())
                ptx::st_release_cta(mailbox + kMbReady, 1);
            __syncwarp();
            int num_taken_chunks = 0;
            while (true) {
                int posted = 0;
                if (lane_idx == 0) {
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                        const int fwd_done = ptx::ld_acquire_cta(mailbox + kMbFwdDone);
                        posted = ptx::ld_acquire_cta(mailbox + kMbPosted);
                        if (posted != num_taken_chunks or fwd_done != 0)
                            return true;
                        if (is_last_check)
                            printf("DeepEP hybrid dispatch (forward helper wait) timeout, channel: %d, taken: %d\n",
                                   channel_idx, num_taken_chunks);
                        return false;
                    });
                }
                posted = __shfl_sync(0xffffffffu, posted, 0);
                __syncwarp();
                if (posted == num_taken_chunks)
                    break;

                const int num_chunk_tokens = mailbox[kMbNumTokens];
                const int fwd_share = mailbox[kMbFwdShare];
                const auto ptrs = reinterpret_cast<void* const*>(mailbox + kMbPtrs);
                const auto src_token_ptr = static_cast<int8_t*>(ptrs[0]);
                const auto chunk_metadata_ptr = static_cast<int*>(ptrs[1]);
                const auto chunk_dst_slot_idx_ptr = static_cast<int*>(ptrs[2]);
                constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
                const auto issue_help_load = [&](const int& b, const int& i) __attribute__((always_inline)) {
                    if (ptx::elect_one_sync()) {
                        const auto mb = b ? mbarrier_ptr_alt : mbarrier_ptr;
                        ptx::tma_load_1d((b ? tma_buffer_alt : tma_buffer).get_base_ptr(),
                                         src_token_ptr + static_cast<int64_t>(i) * scaleout_token_layout.get_num_bytes<false>(),
                                         mb, scaleout_token_layout.get_num_bytes<false>());
                        ptx::mbarrier_arrive_and_set_tx(mb, scaleout_token_layout.get_num_bytes<false>());
                    }
                    __syncwarp();
                };
                ptx::tma_store_wait_read();
                __syncwarp();
                issue_help_load(0, fwd_share);
                for (int i = fwd_share; i < num_chunk_tokens; ++ i) {
                    const int cur = kHelperDoubleBuffer ? (i - fwd_share) & 1 : 0;
                    if (ptx::elect_one_sync()) {
                        if (cur)
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr_alt, phase_alt);
                        else
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }
                    __syncwarp();
                    if constexpr (kHelperDoubleBuffer) {
                        if (i + 1 < num_chunk_tokens) {
                            ptx::tma_store_wait_read();
                            __syncwarp();
                            issue_help_load(cur ^ 1, i + 1);
                        }
                    }
                    forward_token(cur ? tma_buffer_alt : tma_buffer, i, chunk_metadata_ptr + i * kNumForwardMetadataDims,
                                  chunk_dst_slot_idx_ptr + i * kNumTopk, i == num_chunk_tokens - 1);
                    if constexpr (not kHelperDoubleBuffer) {
                        if (i + 1 < num_chunk_tokens) {
                            ptx::tma_store_wait_read();
                            __syncwarp();
                            issue_help_load(0, i + 1);
                        }
                    }
                }
                num_taken_chunks = posted;
                __syncwarp();
                if (ptx::elect_one_sync())
                    ptx::st_release_cta(mailbox + kMbDone, num_taken_chunks);
                __syncwarp();
            }

            // The helper's NVLink stores are complete: count it done on every scale-up peer, next to the
            // forward warp of its channel
            // Each lane waits for its own bulk stores, the warp barrier orders them before every lane's
            // release add
            if constexpr (kNumScaleupRanks > 1) {
                ptx::tma_store_commit();
                ptx::tma_store_wait();
                __syncwarp();
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    if (const int i = j * 32 + lane_idx; i < kNumScaleupRanks) {
                        ptx::red_add_rel_sys(gin.get_sym_ptr<ncclTeamTagLsa>(
                            workspace_layout.get_dispatch_done_counter_ptr(scaleup_rank_idx), i), 1);
                    }
                }
                __syncwarp();
            }
        }

        comm::flush_with_timeout<kNumTimeoutCycles>(gin.gin, ncclCoopWarp());

    } else {
        const int forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);
        const int channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;
        int64_t rx_part_base[kNumParts];
        #pragma unroll
        for (int p = 0; p < kNumParts; ++ p)
            rx_part_base[p] = static_cast<int64_t>(*gin.gin.getSignalShadowPtr(part_signal_id(p)));
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumSlotsPerChannel>(channel_idx);
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        const auto read_batch_header = [&](const int& src, const int& part_idx, const int& sub_idx) -> int64_t {
            return ptx::ld_acquire_sys<int64_t>(
                scaleout_recv_buffer.get_rank_buffer(src)
                    .get_token_buffer(sub_slot(part_idx, sub_idx)).get_hdr_ptr());
        };

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // Shape of `dst_buffer_slot_idx`: `[kNumChannels, kNumScaleoutRanks, kNumMaxTokensPerChannel, kNumTopk]`
        dst_buffer_slot_idx += channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel * kNumTopk);

        // Transform linked list index
        const auto transform_linked_list_idx = [=](const int& idx) {
            constexpr int kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1;
            return channel_idx * (kNumTokensInLinkedList * kNumScaleupRanks) +
                idx * kNumScaleupRanks + scaleup_rank_idx;
        };

        // Forward tokens from scale-out ranks
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Too many scale-out ranks");
        int num_tokens_processed = 0;
        int stored_scaleout_old_tail_idx = 0;
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};
        int stored_finish_flag = lane_idx >= kNumScaleoutRanks;
        int stored_scaleout_tail_idx = 0;
        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;
        int stored_terminal_part = -1;
        int stored_terminal_sub = -1;
        int anomaly_part = -1, anomaly_num_valid = 0;
        bool anomaly_detected = false;
        int poll_start_part = 0;
        int num_posted_chunks = 0;
        // Rows recorded from the own and from the remote sources. The combine replays the recorded order and
        // sends the remote-source rows back over the network. A rising first part reaches the forwarder in small
        // pieces, so there a local turn may run at most `kPaceSlack` rows ahead of the mean remote source while a
        // remote source still sends
        constexpr bool kPaceLocalTurns = kShapeFirstPart;
        constexpr int kPaceSlack = 12, kPaceMinTurn = 8;
        int num_local_recorded = 0, num_remote_recorded = 0;
        // The headers of the remote sources are zero only once the entry flag is open
        wait_entry();
        bool entry_open = true;
        uint32_t wip_mask;
        while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {
            // Pick next rank in round-robin
            const auto offset = (recv_scaleout_rank_idx + 1) % kNumScaleoutRanks;
            const auto hi_mask = (wip_mask >> offset) << offset;
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);

            // Wait for this rank to have data (or finish). The combine replays the recorded
            // forward order, so the sources stay interleaved per channel.
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                const uint32_t arrived_or_finished =
                    stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))
                    return true;

                const bool src_valid = lane_idx < kNumScaleoutRanks;
                const bool is_local_src = (lane_idx == scaleout_rank_idx);
                int intended_finish = 0, intended_tail = 0;
                if (is_local_src) {
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                        workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx));
                    math::unpack2<int, int64_t>(signaled_tail, intended_finish, intended_tail);
                }
                const bool is_remote_src = src_valid and not is_local_src;
                int incremental_tail = stored_scaleout_tail_idx;
                bool prefix_alive = is_remote_src;
                bool my_finished = false;
                int missing_part = -1, missing_num_valid = 0, missing_landed = 0;
                int first_open_part = kNumParts;
                // Parts below `poll_start_part` have passed for every source still sending
                #pragma unroll
                for (int k = 0; k < kNumParts; ++ k) {
                    if (k < poll_start_part)
                        continue;
                    const int64_t landed = ptx::exchange(
                        comm::signal_delta(gin.gin.readSignal(part_signal_id(k)), rx_part_base[k]), 0);

                    const int num_subs = num_subs_of(k);
                    int sub_count[kNumMaxSubParts];
                    bool sub_more[kNumMaxSubParts];
                    bool sub_valid[kNumMaxSubParts];
                    int num_valid = 0;
                    #pragma unroll
                    for (int s = 0; s < kNumMaxSubParts; ++ s) {
                        sub_count[s] = 0;
                        sub_more[s] = false;
                        sub_valid[s] = false;
                        if (s < num_subs) {
                            sub_valid[s] = is_remote_src and
                                unpack_scaleout_header(read_batch_header(lane_idx, k, s),
                                                       sub_count[s], sub_more[s]);
                            num_valid += __popc(ptx::gather(sub_valid[s]));
                        }
                    }
                    const bool part_landed = landed == static_cast<int64_t>(num_valid);
                    if (landed > static_cast<int64_t>(num_valid) and not anomaly_detected) {
                        anomaly_detected = true;
                        anomaly_part = k;
                        anomaly_num_valid = num_valid;
                    }
                    if (static_cast<int64_t>(num_valid) > landed and missing_part < 0) {
                        missing_part = k;
                        missing_num_valid = num_valid;
                        missing_landed = static_cast<int>(landed);
                    }

                    #pragma unroll
                    for (int s = 0; s < kNumMaxSubParts; ++ s) {
                        if (s >= num_subs)
                            continue;
                        const bool through = prefix_alive and sub_valid[s] and part_landed;
                        if (through)
                            incremental_tail = sub_slot(k, s) + sub_count[s];
                        if (through and not sub_more[s]) {
                            my_finished = true;
                            stored_terminal_part = k;
                            stored_terminal_sub = s;
                        }
                        if (prefix_alive and not through and first_open_part == kNumParts)
                            first_open_part = k;
                        prefix_alive = through;
                    }

                    // No source can pass a later part before this one
                    if (not __any_sync(0xffffffffu, prefix_alive))
                        break;
                }
                const bool src_done = my_finished or stored_finish_flag > 0;
                poll_start_part = __reduce_min_sync(0xffffffffu,
                    (is_remote_src and not src_done) ? first_open_part : kNumParts);
                if (is_local_src) {
                    stored_finish_flag       = intended_finish;
                    stored_scaleout_tail_idx = intended_tail;
                } else if (src_valid) {
                    if (incremental_tail > stored_scaleout_tail_idx)
                        stored_scaleout_tail_idx = incremental_tail;
                    if (my_finished)
                        stored_finish_flag = 1;
                }
                __syncwarp();

                // Timeout
                if (is_last_check and lane_idx < kNumScaleoutRanks)
                    printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                           "channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d), "
                           "arrival anomaly: %d, part: %d, num valid: %d, "
                           "completion missing: %d, missing part: %d, valid headers: %d, completions: %d\n",
                           scaleout_rank_idx, scaleup_rank_idx,
                           channel_idx, lane_idx, stored_scaleout_old_tail_idx,
                           stored_finish_flag, stored_scaleout_tail_idx,
                           static_cast<int>(anomaly_detected), anomaly_part, anomaly_num_valid,
                           static_cast<int>(missing_part >= 0), missing_part, missing_num_valid, missing_landed);
                return false;
            });

            // NVLink writes into peers wait for the entry flag
            if (not entry_open) {
                wait_entry();
                entry_open = true;
            }

            // Process one chunk from the current rank
            const auto start_slot_idx = ptx::exchange(stored_scaleout_old_tail_idx, recv_scaleout_rank_idx);
            int turn_limit = kNumSlotsPerForwardChunk;
            if (kPaceLocalTurns and kNumScaleoutRanks > 1 and recv_scaleout_rank_idx == scaleout_rank_idx and
                (wip_mask & ~(1u << scaleout_rank_idx)) != 0) {
                const int lag = num_remote_recorded / (kNumScaleoutRanks - 1) + kPaceSlack - num_local_recorded;
                turn_limit = lag < kPaceMinTurn ? kPaceMinTurn : (lag < kNumSlotsPerForwardChunk ? lag : kNumSlotsPerForwardChunk);
            }
            const auto end_slot_idx = std::min(
                ptx::exchange(stored_scaleout_tail_idx, recv_scaleout_rank_idx),
                start_slot_idx + turn_limit
            );
            if (lane_idx == recv_scaleout_rank_idx)
                stored_scaleout_old_tail_idx = end_slot_idx;
            if (recv_scaleout_rank_idx == scaleout_rank_idx)
                num_local_recorded += end_slot_idx - start_slot_idx;
            else
                num_remote_recorded += end_slot_idx - start_slot_idx;

            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);

            const layout::TokenLayout fwd_bufs[2] = {tma_buffer, tma_buffer_alt};
            ptx::mbarrier* const fwd_mbs[2] = {mbarrier_ptr, mbarrier_ptr_alt};
            ptx::arrival_phase* const fwd_phases[2] = {&phase, &phase_alt};
            const auto issue_fwd_load = [&](const int& b, const int& slot) {
                const auto tb = recv_buffer.get_token_buffer(slot);
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(fwd_bufs[b].get_base_ptr(), tb.get_base_ptr(),
                                     fwd_mbs[b], scaleout_token_layout.get_num_bytes<false>());
                    ptx::mbarrier_arrive_and_set_tx(fwd_mbs[b], scaleout_token_layout.get_num_bytes<false>());
                }
                __syncwarp();
            };

            if constexpr (kChunkRouting) {
                // A buffer is reloaded once its stores have read it, the exit barrier waits for completion
                ptx::tma_store_wait_read();
                __syncwarp();
                if (start_slot_idx < end_slot_idx)
                    issue_fwd_load(0, start_slot_idx);

                // The helper reads the tables of the chunk it took until it reports it done
                if (num_posted_chunks > 0 and start_slot_idx < end_slot_idx) {
                    if (ptx::elect_one_sync()) {
                        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                            if (ptx::ld_acquire_cta(mailbox + kMbDone) == num_posted_chunks)
                                return true;
                            if (is_last_check)
                                printf("DeepEP hybrid dispatch (forward helper) timeout, channel: %d, posted: %d, done: %d\n",
                                       channel_idx, num_posted_chunks, ptx::ld_acquire_cta(mailbox + kMbDone));
                            return false;
                        });
                    }
                    __syncwarp();
                }

                // Route the chunk while its first token loads, one token per lane: scale-up ranks, slots
                // (one slot atomic per scale-up rank and chunk) and linked-list indices
                using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
                EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Invalid number of scale-up peers");
                if (start_slot_idx < end_slot_idx) {
                    const bool routes = lane_idx < end_slot_idx - start_slot_idx;
                    const auto routed = recv_buffer.get_token_buffer(start_slot_idx + (routes ? lane_idx : 0));
                    constexpr int kNumTopkVecs = math::constexpr_ceil_div(kNumTopk * static_cast<int>(sizeof(int16_t)),
                                                                          static_cast<int>(sizeof(int4)));
                    int4 topk_vecs[kNumTopkVecs];
                    #pragma unroll
                    for (int i = 0; i < kNumTopkVecs; ++ i)
                        topk_vecs[i] = ptx::ld_with_gez_pred(
                            reinterpret_cast<const int4*>(routed.get_scaleout_topk_idx_ptr()) + i, routes ? 0 : -1);
                    int cached_slots[kNumTopk];
                    if constexpr (kReuseSlotIndices) {
                        const auto cached_slot_ptr = dst_buffer_slot_idx +
                            recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) +
                            (start_slot_idx + (routes ? lane_idx : 0)) * kNumTopk;
                        #pragma unroll
                        for (int k = 0; k < kNumTopk; ++ k)
                            cached_slots[k] = routes ? __ldg(cached_slot_ptr + k) : -1;
                    }

                    const auto topk16 = reinterpret_cast<const int16_t*>(topk_vecs);
                    int ranks[kNumTopk];
                    mask_t rank_mask = 0;
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        const int expert_idx = (routes ? static_cast<int>(topk16[k]) : -1) - scaleout_rank_idx * kNumExpertsPerScaleout;
                        ranks[k] = 0 <= expert_idx and expert_idx < kNumExpertsPerScaleout ? expert_idx / kNumExpertsPerRank : -1;
                        rank_mask |= ranks[k] >= 0 ? (mask_t(1) << ranks[k]) : mask_t(0);
                    }

                    int chunk_counts[kNumScaleupRanksPerLane] = {};
                    const unsigned lanes_below = (1u << lane_idx) - 1u;
                    #pragma unroll
                    for (int r = 0; r < kNumScaleupRanks; ++ r) {
                        const unsigned tokens_to_r = ptx::gather((rank_mask >> r) & 1);
                        if (routes)
                            route_prefix[lane_idx * kNumScaleupRanks + r] = __popc(tokens_to_r & lanes_below);
                        if (lane_idx == r % 32)
                            chunk_counts[r / 32] = __popc(tokens_to_r);
                    }
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        if (const int r = j * 32 + lane_idx; r < kNumScaleupRanks) {
                            if constexpr (not kReuseSlotIndices) {
                                route_base[r] = chunk_counts[j] > 0 ?
                                    atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + r, chunk_counts[j]) : 0;
                            }
                            route_count[r] = stored_scaleup_send_counters[j];
                            stored_scaleup_send_counters[j] += chunk_counts[j];
                        }
                    }
                    __syncwarp();

                    // The highest top-k lane of a scale-up rank owns its slot, as `ptx::deduplicate` picks
                    if (routes) {
                        mask_t owned = 0;
                        #pragma unroll
                        for (int k = kNumTopk - 1; k >= 0; -- k) {
                            const int r = ranks[k];
                            int slot = kReuseSlotIndices ? cached_slots[k] : -1, ll = -1;
                            if (r >= 0) {
                                const int prefix = route_prefix[lane_idx * kNumScaleupRanks + r];
                                ll = route_count[r] + prefix;
                                if (not kReuseSlotIndices and not ((owned >> r) & 1))
                                    slot = route_base[r] + prefix;
                                owned |= mask_t(1) << r;
                            }
                            route_rank[lane_idx * kNumTopk + k] = r;
                            route_slot[lane_idx * kNumTopk + k] = slot;
                            route_ll[lane_idx * kNumTopk + k] = ll;
                        }
                    }
                    __syncwarp();
                }

                // Hand the second half of the chunk to the helper when it is ready
                int fwd_end_slot_idx = end_slot_idx;
                const int num_chunk_tokens = end_slot_idx - start_slot_idx;
                const auto chunk_metadata_ptr = token_metadata_at_forward + num_tokens_processed * kNumForwardMetadataDims;
                const auto chunk_dst_slot_idx_ptr = dst_buffer_slot_idx +
                    recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) + start_slot_idx * kNumTopk;
                const bool helper_ready = num_chunk_tokens >= 2 and
                    __shfl_sync(0xffffffffu, lane_idx == 0 ? ptx::ld_acquire_cta(mailbox + kMbReady) : 0, 0) != 0;
                if (helper_ready) {
                    const int fwd_share = (num_chunk_tokens + 1) / 2;
                    fwd_end_slot_idx = start_slot_idx + fwd_share;
                    if (ptx::elect_one_sync()) {
                        mailbox[kMbNumTokens] = num_chunk_tokens;
                        mailbox[kMbFwdShare] = fwd_share;
                        auto ptrs = reinterpret_cast<void**>(mailbox + kMbPtrs);
                        ptrs[0] = recv_buffer.get_token_buffer(start_slot_idx).get_base_ptr();
                        ptrs[1] = chunk_metadata_ptr;
                        ptrs[2] = chunk_dst_slot_idx_ptr;
                        __threadfence_block();
                        ptx::st_release_cta(mailbox + kMbPosted, num_posted_chunks + 1);
                    }
                    num_posted_chunks += 1;
                    __syncwarp();
                }

                for (int slot_idx = start_slot_idx; slot_idx < fwd_end_slot_idx; ++ slot_idx) {
                    const int cur = (slot_idx - start_slot_idx) & 1;
                    const auto& tma_buffer = fwd_bufs[kDoubleBufferForward ? cur : 0];

                    if constexpr (kDoubleBufferForward) {
                        // Wait for this slot's load to complete.
                        if (ptx::elect_one_sync())
                            ptx::mbarrier_wait_and_flip_phase(fwd_mbs[cur], *fwd_phases[cur]);
                        __syncwarp();

                        if (slot_idx + 1 < fwd_end_slot_idx) {
                            ptx::tma_store_wait_read();
                            __syncwarp();
                            issue_fwd_load(cur ^ 1, slot_idx + 1);
                        }
                    } else {
                        // Single buffer: the chunk's first load was issued before the routing
                        if (slot_idx > start_slot_idx) {
                            ptx::tma_store_wait_read();
                            __syncwarp();
                            issue_fwd_load(0, slot_idx);
                        }
                        if (ptx::elect_one_sync())
                            ptx::mbarrier_wait_and_flip_phase(fwd_mbs[0], *fwd_phases[0]);
                        __syncwarp();
                    }

                    const int i = slot_idx - start_slot_idx;
                    forward_token(tma_buffer, i, chunk_metadata_ptr + i * kNumForwardMetadataDims,
                                  chunk_dst_slot_idx_ptr + i * kNumTopk, slot_idx == end_slot_idx - 1);
                }
                num_tokens_processed += num_chunk_tokens;
            } else {
                // A buffer is reloaded once its stores have read it, the exit barrier waits for completion
                if constexpr (kDoubleBufferForward) {
                    ptx::tma_store_wait_read();
                    __syncwarp();
                    if (start_slot_idx < end_slot_idx)
                        issue_fwd_load(0, start_slot_idx);
                }

                for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {
                    const int cur = (slot_idx - start_slot_idx) & 1;
                    const auto& tma_buffer = fwd_bufs[kDoubleBufferForward ? cur : 0];

                    if constexpr (kDoubleBufferForward) {
                        // Wait for this slot's load to complete.
                        if (ptx::elect_one_sync())
                            ptx::mbarrier_wait_and_flip_phase(fwd_mbs[cur], *fwd_phases[cur]);
                        __syncwarp();

                        if (slot_idx + 1 < end_slot_idx) {
                            ptx::tma_store_wait_read();
                            __syncwarp();
                            issue_fwd_load(cur ^ 1, slot_idx + 1);
                        }
                    } else {
                        // Original single-buffer path: load this slot synchronously.
                        ptx::tma_store_wait_read();
                        __syncwarp();
                        issue_fwd_load(0, slot_idx);
                        if (ptx::elect_one_sync())
                            ptx::mbarrier_wait_and_flip_phase(fwd_mbs[0], *fwd_phases[0]);
                        __syncwarp();
                    }

                    // Read the received top-k indices, then widen them and move the weights and the
                    // source index to their scale-up offsets
                    EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
                    int stored_dst_scaleup_rank_idx = -1;
                    const auto received = layout::TokenLayout(
                        kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true, tma_buffer.get_base_ptr(), true);
                    int dst_expert_idx = -1;
                    float topk_weight = 0;
                    if (lane_idx < kNumTopk) {
                        dst_expert_idx = received.get_scaleout_topk_idx_ptr()[lane_idx];
                        topk_weight = received.get_topk_weights_ptr()[lane_idx];
                    }
                    const int src_token_global_idx = *received.get_src_token_global_idx_ptr();
                    __syncwarp();
                    if (lane_idx < kNumTopk) {
                        tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                        tma_buffer.get_topk_weights_ptr()[lane_idx] = topk_weight;
                    }
                    if (lane_idx == 0)
                        *tma_buffer.get_src_token_global_idx_ptr() = src_token_global_idx;
                    ptx::tma_store_fence();
                    __syncwarp();
                    dst_expert_idx -= scaleout_rank_idx * kNumExpertsPerScaleout;
                    stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < kNumExpertsPerScaleout ?
                        dst_expert_idx / kNumExpertsPerRank : -1;

                    // Write the per-scaleup channel index for this token
                    int linked_list_idx = -1;
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;
                        const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;
                        const auto exchanged = ptx::exchange(
                            stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                        linked_list_idx = valid ? exchanged : linked_list_idx;
                    }
                    if (not kReuseSlotIndices and lane_idx < kNumTopk) {
                        tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                        ptx::tma_store_fence();
                    }
                    __syncwarp();

                    // Deduplicate for scale-up ranks
                    int stored_dst_slot_idx = -1;
                    const auto dst_slot_idx_ptr = dst_buffer_slot_idx +
                        recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) + slot_idx * kNumTopk;
                    if constexpr (kReuseSlotIndices) {
                        if (lane_idx < kNumTopk)
                            stored_dst_slot_idx = __ldg(dst_slot_idx_ptr + lane_idx);
                    } else {
                        // Deduplicate for NVLink ranks
                        if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)
                            stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                    }
                    __syncwarp();

                    // Issue TMAs
                    if (stored_dst_slot_idx >= 0) {
                        const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                            scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                            stored_dst_scaleup_rank_idx);
                        ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                        ptx::tma_store_commit();
                    }
                    __syncwarp();

                    // Add per-scale-up counter
                    EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Invalid number of scale-up peers");
                    using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
                    const auto scaleup_send_mask = ptx::reduce_or(
                        stored_dst_scaleup_rank_idx >= 0 ?
                        (mask_t(1) << stored_dst_scaleup_rank_idx) : mask_t(0));
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                        stored_scaleup_send_counters[j] += (scaleup_send_mask >> (j * 32 + lane_idx)) & 1;

                    // Record metadata at forward
                    if constexpr (not kReuseSlotIndices) {
                        EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of selections");
                        const auto metadata_ptr = token_metadata_at_forward +
                            num_tokens_processed * kNumForwardMetadataDims;

                        // Source token index and last token index flag
                        if (ptx::elect_one_sync()) {
                            metadata_ptr[0] = tma_buffer.get_src_token_global_idx_ptr()[0];
                            metadata_ptr[1] = slot_idx == (end_slot_idx - 1);
                        }

                        // Second, original top-k indices and destination slots
                        if (lane_idx < kNumTopk) {
                            metadata_ptr[2 + lane_idx] = stored_dst_scaleup_rank_idx;
                            metadata_ptr[2 + kNumTopk + lane_idx] = stored_dst_slot_idx;
                            dst_slot_idx_ptr[lane_idx] = stored_dst_slot_idx;
                        }
                    }
                    num_tokens_processed += 1;
                    __syncwarp();
                }
            }
        }

        // The helper leaves once every posted chunk is taken
        if constexpr (kChunkRouting) {
            if (ptx::elect_one_sync())
                ptx::st_release_cta(mailbox + kMbFwdDone, 1);
            __syncwarp();
        }

        // Assign the source token index part of the metadata into `-1` as an ending mark
        if (not kReuseSlotIndices and ptx::elect_one_sync())
            token_metadata_at_forward[num_tokens_processed * kNumForwardMetadataDims] = -1;
        __syncwarp();

        if (not entry_open)
            wait_entry();

        // Update linked list's ending position
        if constexpr (not kReuseSlotIndices) {
            const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                    ptx::st_relaxed_sys(
                        gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j),
                        transform_linked_list_idx(stored_scaleup_send_counters[i]));
                }
            }
        }
        __syncwarp();

        // Lane k drains part k, all parts at once (one signal read round trip instead of one per part)
        EP_STATIC_ASSERT(kNumParts <= 32, "One lane drains one part");
        int drain_expected = 0;
        int64_t drain_base = 0;
        #pragma unroll
        for (int k = 0; k < kNumParts; ++ k) {
            const int num_subs_k = num_subs_of(k);
            const int my_subs_k = stored_terminal_part > k  ? num_subs_k
                                : stored_terminal_part == k ? stored_terminal_sub + 1
                                                            : 0;
            const int expected_k = ptx::reduce_add(my_subs_k);
            drain_expected = lane_idx == k ? expected_k : drain_expected;
            drain_base = lane_idx == k ? rx_part_base[k] : drain_base;
        }
        const bool drains = lane_idx < kNumParts and drain_expected > 0;
        comm::timeout_while<kNumTimeoutCycles>(drains, [&](const bool& is_last_check) {
            // Only the count matters here (the data was consumed with acquires in the poll), so a relaxed read
            const int64_t landed = comm::signal_delta(
                gin.gin.readSignal(part_signal_id(lane_idx), 64, cuda::memory_order_relaxed), drain_base);
            if (landed >= static_cast<int64_t>(drain_expected))
                return true;
            if (is_last_check)
                printf("DeepEP hybrid dispatch (epilogue shadow drain) timeout, "
                       "channel: %d, part: %d, landed: %lld, expected: %d\n",
                       channel_idx, lane_idx, (long long)landed, drain_expected);
            return false;
        });
        if (drains)
            gin.gin.increaseSignalShadow(part_signal_id(lane_idx), static_cast<uint64_t>(drain_expected));
        __syncwarp();

        if (lane_idx < kNumScaleoutRanks)
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
        __syncwarp();

        // This channel's NVLink stores and tails are complete: count it done on every scale-up peer
        // (each lane waits for its own bulk stores, the warp barrier orders them before every lane's release add)
        if constexpr (kNumScaleupRanks > 1) {
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            __syncwarp();
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                if (const int i = j * 32 + lane_idx; i < kNumScaleupRanks) {
                    ptx::red_add_rel_sys(gin.get_sym_ptr<ncclTeamTagLsa>(
                        workspace_layout.get_dispatch_done_counter_ptr(scaleup_rank_idx), i), 1);
                }
            }
            __syncwarp();
        }
    }

    // The copy epilogue waits for the peers' done counters instead of an exit barrier. Each SM counts
    // itself out, and the last one resets the slot counters and the entry flag (every warp of every SM
    // has passed its entry wait and its slot atomics) and moves the done target by one dispatch.
    if constexpr (kNumScaleupRanks > 1) {
        ptx::tma_store_commit();
        ptx::tma_store_wait();
        __syncthreads();
        cudaTriggerProgrammaticLaunchCompletion();
        if (thread_idx == 0) {
            __threadfence();
            const auto exit_counter_ptr = workspace_layout.get_dispatch_exit_counter_ptr();
            if (atomicAdd(exit_counter_ptr, 1) == kNumSMs - 1) {
                __threadfence();
                if constexpr (not kReuseSlotIndices) {
                    #pragma unroll
                    for (int i = 0; i < kNumScaleupRanks; ++ i)
                        workspace_layout.get_scaleup_atomic_sender_counter()[i] = 0;
                }
                *workspace_layout.get_dispatch_done_target_ptr() += kNumChannels * (kChunkRouting ? 2 : 1);
                *entry_flag_ptr = 0;
                *exit_counter_ptr = 0;
            }
        }
    } else {
        comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                          kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag1, true, true, false>(
            gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, /* do not scale-out */ false, true);
        cudaTriggerProgrammaticLaunchCompletion();
        if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumScaleupRanks)
            workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
        if (sm_idx == 0 and thread_idx == 0)
            *entry_flag_ptr = 0;
    }
}

}  // namespace deep_ep::elastic
