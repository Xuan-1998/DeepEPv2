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
#include <deep_ep/common/gin_resource_alloc.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>
#include <deep_ep/impls/proxy_ring.cuh>

#ifndef EP_COMBINE_BATCH_SIZE
#define EP_COMBINE_BATCH_SIZE 6
#endif

namespace deep_ep::elastic {

// Small batch: tokens per scale-out put batch. The receiver derives its expected signal count
// from the same constant. Small batches let the puts start while the forwarders are still
// reducing; each put costs the proxy warp about 12 us, so 2 is the measured optimum at 128
// tokens (1 floods the proxies, 12 posts everything at the end).
#ifndef EP_SB_BATCH_SIZE
#define EP_SB_BATCH_SIZE 2
#endif
#ifndef EP_SB_PROBE_EVERY
#define EP_SB_PROBE_EVERY 64
#endif
// Small batch: push and reduce the tokens of remote sources before the local ones, so the
// scale-out puts start as early as possible and the local-source work runs under the RDMA
#ifndef EP_SB_REMOTE_FIRST
#define EP_SB_REMOTE_FIRST 1
#endif
// Small batch proxy: 0 = one warp per channel posting its batches one by one, 1 = one warp per
// SM collecting the batches of all channels and posting them together (one coalesced post per QP)
#ifndef EP_SB_PROXY_MODE
#define EP_SB_PROXY_MODE 0
#endif
// Small batch: prefetch the expert outputs into L2 at kernel start (forward warps, which wait for
// the first partials anyway), so the scale-up warps' bulk loads hit L2 instead of HBM
#ifndef EP_SB_L2_PREFETCH
#define EP_SB_L2_PREFETCH 0
#endif
// Small batch: publish a scale-up step's tails at the next token, after its bulk load has been
// issued, so waiting for the step's stores to complete overlaps that load instead of stalling
#ifndef EP_SB_DEFER_TAILS
#define EP_SB_DEFER_TAILS 0
#endif
// Small batch: publish the scale-up tails every this many linked-list steps (and at the end of
// each remote-first pass). The system-scope release before each publication costs a few
// microseconds, more than pushing the step itself, so fewer publications shorten every warp's chain
#ifndef EP_SB_TAIL_EVERY
#define EP_SB_TAIL_EVERY 1
#endif
// Diagnostic: publish the tails with relaxed stores after the bulk stores completed (no system
// release fence). Only for measuring what the release costs; not a correct protocol by itself
#ifndef EP_SB_RELAXED_TAILS
#define EP_SB_RELAXED_TAILS 0
#endif
// Small batch: scale-up warps hand their per-peer counts to a publisher warp through shared memory
// (CTA-scope release) instead of publishing the tails themselves. The publisher batches every
// pending tail behind one system-scope fence, so the sending warps never wait for that fence.
// Needs proxy mode 1 (it runs on a proxy warp slot that mode leaves idle)
#ifndef EP_SB_TAIL_PUBLISHER
#define EP_SB_TAIL_PUBLISHER 0
#endif
// Small batch: scale-up warps prefetch the rows of a linked-list step into L2 when they read the
// step (lane-parallel), so the step's per-peer bulk loads after the first one hit L2
#ifndef EP_SB_SU_PREFETCH
#define EP_SB_SU_PREFETCH 0
#endif
// Small batch: unroll factors (vectors per lane per step, capped by what divides the hidden size)
// of the scale-up warps' local reduce and of the forwarders' reduce in the expanded layout, where
// the register adjustment is off. Each step of the reduce is one memory latency
#ifndef EP_SB_SU_UNROLL
#define EP_SB_SU_UNROLL 4
#endif
#ifndef EP_SB_FWD_UNROLL_EX
#define EP_SB_FWD_UNROLL_EX 4
#endif
// Small batch, plain layout with the register move: vectors per lane per reduce step in the
// forwarders. Each step loads this many vectors from every unconditional source at once, so the
// step's live registers are (sources + 1) x 4 x this value
#ifndef EP_SB_FWD_UNROLL
#define EP_SB_FWD_UNROLL 8
#endif
// Small batch, 4 channels per SM: forward-warp registers in the plain layout (the scale-up warps
// release enough for up to 248), how many sources the forwarders' reduce loads together, and a
// register move for the expanded layout (proxy warpgroup down to 64, forward warps up to 192)
#ifndef EP_SB_FWD_REGS
#define EP_SB_FWD_REGS 216
#endif
#ifndef EP_SB_FWD_HOIST
#define EP_SB_FWD_HOIST 0
#endif
#ifndef EP_SB_EX_REGS
#define EP_SB_EX_REGS 0
#endif
// Small batch, expanded layout: sources the scale-up warps' local reduce loads together before
// adding (the default of one makes every extra source of a row cost a full memory latency per
// chunk; a missing source is a predicated-off load and costs nothing)
#ifndef EP_SB_SU_HOIST
#define EP_SB_SU_HOIST 0
#endif
// Small batch: forwarders load the merged top-k weights before the reduce instead of after it
#ifndef EP_SB_EARLY_WEIGHTS
#define EP_SB_EARLY_WEIGHTS 0
#endif
// Small batch: the proxy warp flushes its GIN contexts before it exits, so no put of this call
// can still read the send buffer when the kernel returns. Without it, the next writer of these
// send slots (the combine two calls later) is only ordered after this call's puts when this
// rank receives data from the same peers in between
#ifndef EP_SB_COMBINE_FLUSH
#define EP_SB_COMBINE_FLUSH 0
#endif
// Small batch with the remote-first order: scale-up warps reorder each peer's linked list once
// (remote-source entries first) and push round by round, the r-th remote row of every peer in
// round r. The list follows the dispatch's forwarding order, which puts local tokens first, so
// walking it in lockstep steps pushed a peer's first remote row only after other peers' rows
#ifndef EP_SB_SU_COMPACT
#define EP_SB_SU_COMPACT 0
#endif
// Small batch with the tail publisher: in the remote pass, hand each row's count to the
// publisher right after its bulk store (1) or once the next row's load is in flight (2),
// instead of once per linked-list step
#ifndef EP_SB_SU_ROW_TAILS
#define EP_SB_SU_ROW_TAILS 0
#endif
// Small batch: device-scope fence before a forwarder hands a batch to the proxy
#ifndef EP_SB_HANDOFF_FENCE
#define EP_SB_HANDOFF_FENCE 1
#endif
// Small batch: forwarders reduce straight into the scale-out send slot (or the local receive
// slot) in global memory instead of staging the token in shared memory and bulk-storing it
#ifndef EP_SB_FWD_DIRECT
#define EP_SB_FWD_DIRECT 0
#endif
// Small batch, expanded layout with the compact walk: scale-up warps load the top-k slots of a
// whole round of rows at once (the per-row load sat between knowing the row and its reduce)
#ifndef EP_SB_SU_ROUND_SLOTS
#define EP_SB_SU_ROUND_SLOTS 0
#endif
// Small batch with the remote-first order: the scale-up warps of an SM wait for each other
// (named barrier 1) between their remote and their local rows
#ifndef EP_SB_SU_PASS_BARRIER
#define EP_SB_SU_PASS_BARRIER 0
#endif
// Small batch, proxy mode 1: post every scale-out batch as this many puts of equal size. Puts
// above one EFA packet (about 8 KB) take far longer to complete than single-packet ones, and the
// receiver counts every piece (each one carries the signal)
#ifndef EP_SB_PUT_SPLIT
#define EP_SB_PUT_SPLIT 1
#endif
// Small batch, proxy mode 1: size the scale-out pieces per token instead of per batch (this many
// pieces per token, 0 = use `EP_SB_PUT_SPLIT`), so a batch shorter than `EP_SB_BATCH_SIZE`
// posts fewer, still single-packet puts
#ifndef EP_SB_PIECES_PER_TOKEN
#define EP_SB_PIECES_PER_TOKEN 0
#endif
// Small batch, proxy mode 1: proxy-slot warps per SM (0 = one per channel). Mode 1 uses two
// (the proxy and the tail publisher), so the others only take registers from the data warps.
// The host reads the same define from `EP_JIT_EXTRA_FLAGS` to size the block
#ifndef EP_SB_PROXY_WARPS
#define EP_SB_PROXY_WARPS 0
#endif
// Small batch, proxy mode 1: start a sweep only with at least this many batches, unless every
// forwarder has handed over its last batch
#ifndef EP_SB_PROXY_MIN
#define EP_SB_PROXY_MIN 1
#endif
// Small batch with the remote-first order: forwarders mark their ring done after the remote
// pass (the local pass hands over no batch), so the proxy knows the remote pass is complete
#ifndef EP_SB_EARLY_DONE
#define EP_SB_EARLY_DONE 0
#endif
// Diagnostics (small batch, results are wrong): skip the scale-up data movement, the forwarders'
// tail wait or their reduce, to measure what each costs on the critical path
#ifndef EP_SB_DIAG_NO_SU
#define EP_SB_DIAG_NO_SU 0
#endif
#ifndef EP_SB_DIAG_NO_WAIT
#define EP_SB_DIAG_NO_WAIT 0
#endif
#ifndef EP_SB_DIAG_NO_REDUCE
#define EP_SB_DIAG_NO_REDUCE 0
#endif
// Diagnostic: cap the bytes of every scale-out put (0 = no cap)
#ifndef EP_SB_DIAG_PUT_BYTES
#define EP_SB_DIAG_PUT_BYTES 0
#endif
#ifdef EP_SB_PROBE
// Small-batch combine timeline probes (raw %globaltimer ns, per block), printed by SM 0 for
// two of every EP_SB_PROBE_EVERY iterations. Slots: 0 entry | 1 max scale-up warps done |
// 2 max first tail wait satisfied (forwarders) | 3 max last reduce handed to proxy | 4 max
// proxy put issued | 5 max inbound signal wait satisfied | 6 max proxy warp exit | 7 max tokens
// per scale-up warp | 8 earliest put (inverted) | 9 puts | 10 sum put issue ns | 11 sum pending
// per put | 14 max forward warp exit
// 12 sum of forwarder tail-wait ns | 13 sum of forwarder reduce ns | 15 forwarded tokens |
// 16 publisher rounds | 17 sum of publisher fence ns | 18 max last publication | 19 max end of
// the remote pass (forwarders) | 20 max start of a proxy sweep | 21 earliest end of the remote pass
// (inverted) | 22 remote-pass tokens | 23 remote-pass tail-wait ns | 24 remote-pass reduce ns |
// 25 max end of the scale-up remote pass
__device__ unsigned long long sbc_probe[128][28];
// Per channel (forwarder): 0 remote tokens | 1 remote pass end | 2 remote-pass wait ns |
// 3 remote-pass reduce ns | 4 local tokens | 5 forward exit | 6 first remote token ready.
// Per scale-up warp: 0 remote rows | 1 remote pass end | 2 rows | 3 end
__device__ unsigned long long sbc_chan[512][8];
__device__ unsigned long long sbc_su[1024][4];
#endif

// Iterate the whole forward metadata list and count how many tokens (forward slots) this channel
// will send to each scale-out destination. Warp-wide: each lane scans a stride, then the per-lane
// tallies are summed.
template <bool kAllowMultipleReduction, bool kUseExpandedLayout,
          int kNumScaleoutRanks, int kNumScaleupRanks, int kNumMaxTokensPerRank,
          int kNumMaxTokensPerChannel, int kNumTopk, int kNumForwardMetadataDims>
__device__ __forceinline__ void count_forward_slots_per_dst(
        const int* token_metadata_at_forward, const int& lane_idx,
        int (&out)[kNumScaleoutRanks]) {
    // Two FIFOs (see the dispatch forwarder): remote tokens from entry 0 upward, local tokens from
    // kForwardListTop downward, each ended by -1
    constexpr int kForwardListTop = kNumScaleoutRanks * kNumMaxTokensPerChannel;
    for (int region = 0; region < 2; ++ region)
    for (int base = 0; ; base += 32) {
        const int idx = region == 0 ? base + lane_idx : kForwardListTop - (base + lane_idx);
        const bool in_range = 0 <= idx and idx <= kForwardListTop;
        const int g = in_range
            ? __ldg(token_metadata_at_forward + idx * kNumForwardMetadataDims) : -1;
        // Lanes at or beyond the `-1` sentinel are past the end of the list.
        const unsigned alive = ptx::gather(g >= 0);
        const int first_dead = __ffs(static_cast<int>(~alive));
        const int live_lanes = first_dead == 0 ? 32 : first_dead - 1;
        if (lane_idx < live_lanes) {
            const int dst = (g / kNumMaxTokensPerRank) / kNumScaleupRanks;
            int slots = 1;
            if constexpr (not kAllowMultipleReduction) {
                int n = 0;
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k) {
                    const int v = __ldg(token_metadata_at_forward +
                                        idx * kNumForwardMetadataDims + 2 + k);
                    if (v < 0)
                        continue;
                    bool dup = false;
                    if constexpr (not kUseExpandedLayout) {
                        #pragma unroll
                        for (int pk = 0; pk < k; ++ pk)
                            dup |= (__ldg(token_metadata_at_forward +
                                          idx * kNumForwardMetadataDims + 2 + pk) == v);
                    }
                    n += not dup;
                }
                slots = n;
            }
            out[dst] += slots;
        }
        if (live_lanes < 32)
            break;
    }
    #pragma unroll
    for (int d = 0; d < kNumScaleoutRanks; ++ d)
        out[d] = ptx::reduce_add(out[d]);
}

// Small-batch mode (`kSmallBatch`): the decode-sized variant, paired with the small-batch
// dispatch. No entry barrier: the host alternates the combine buffer between two parity
// halves per combine call (independent of the dispatch halves) and the per-channel scale-up
// tails and inbound signals are parity-indexed, so a peer one call ahead cannot collide with
// this rank's consumption or resets. Each channel runs several scale-up warps (the peers are
// split between them, see `kNumScaleupWarpsPerChannel`) because one warp pushes about 14 KB
// per 3 us over NVLink, and one proxy warp per channel that posts the channel's scale-out
// batches (`EP_SB_BATCH_SIZE` tokens) as soon as they are handed over, so the RDMA overlaps
// the reduce phase. The HT variant (`kSmallBatch == false`) is unchanged.
template <bool kSmallBatch, int kNumGinSignals,
          bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs,
          int kNumScaleupWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumScaleupUpdateInterval = kSmallBatch ? EP_SB_TAIL_EVERY : 3,
          int kBatchSize = kSmallBatch ? EP_SB_BATCH_SIZE : EP_COMBINE_BATCH_SIZE,
          int kProxyRingDepth = kSmallBatch ? kProxyRingDepthSmallBatch : kProxyRingDepthDefault,
          int kNumChannelsPerSM = kNumForwardWarps,
          int kNumChannels = kNumChannelsPerSM * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumDataWarps = kNumScaleupWarps + kNumForwardWarps,
          int kProxyWarpIdx = kNumDataWarps,
          // Small batch: one proxy warp per channel, so the puts of a channel are posted while
          // the others are still reducing (one post takes about 12 us)
          int kNumProxyWarps = kSmallBatch ? (EP_SB_PROXY_WARPS > 0 ? EP_SB_PROXY_WARPS : kNumForwardWarps) : 1,
          int kNumWarps = kNumDataWarps + kNumProxyWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          bool kUseScaleoutRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          bool kUseScaleupRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>(),
          int kNumTokensInScaleoutLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          int kNumTokensInScaleupLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_unordered_combine_impl(nv_bfloat16* x,
                    float* topk_weights,
                    int* src_metadata,
                    int* psum_num_recv_tokens_per_scaleup_rank,
                    int* token_metadata_at_forward,
                    int* channel_linked_list,
                    const int* token_map_at_dispatch,
                    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                    void* buffer, void* workspace,
                    const int scaleout_rank_idx, const int scaleup_rank_idx,
                    int num_reduced_tokens, const int num_combined_tokens,
                    const int combine_iteration) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;
    constexpr bool kRemoteFirst = kSmallBatch and kNumScaleoutRanks > 1 and EP_SB_REMOTE_FIRST != 0;
    constexpr int kNumOrderPasses = kRemoteFirst ? 2 : 1;

    // Small-batch mode: parity of this combine call (buffer half, tail slots, signal ids)
    const int sb_parity = combine_iteration & 1;
    // Small-batch signal ids: the dispatch owns [0, 2 * cpc), the combine [2 * cpc, 4 * cpc),
    // both indexed (channel-in-context, parity). Mirrors `part_signal_id_of` in the dispatch.
    constexpr int kNumSBChannelsPerContext = gin_alloc::channels_per_context(
        kNumSMs, kNumQPs - 1, kNumChannelsPerSM);
    EP_STATIC_ASSERT(not kSmallBatch or 4 * kNumSBChannelsPerContext <= kNumGinSignals,
                     "GIN signal budget cannot host the small-batch dispatch + combine signals");
    const auto channel_signal_id = [&](const int& channel_in_sm_idx) {
        const int qp_signal_id = comm::get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, channel_in_sm_idx);
        if constexpr (kSmallBatch)
            return static_cast<ncclGinSignal_t>(2 * kNumSBChannelsPerContext + qp_signal_id * 2 + sb_parity);
        else
            return static_cast<ncclGinSignal_t>(qp_signal_id);
    };

    // Combine vector type selection
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
    // Per-channel scale-up tails: parity-indexed in small-batch mode, legacy otherwise
    const auto scaleup_tail_ptr = [&](const int& channel_idx, const int& scaleup_rank) {
        if constexpr (kSmallBatch)
            return workspace_layout.get_sb_combine_tail_ptr(sb_parity, channel_idx, scaleup_rank);
        else
            return workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank);
    };

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumScaleupRanks - 1);

    // Token layouts
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

    // Small batch, direct reduce: the forwarders write into global memory and need no staging
    // buffer, so only the scale-up warps get one (the host sizes the arena the same way)
    constexpr bool kFwdDirect = kSmallBatch and kAllowMultipleReduction and EP_SB_FWD_DIRECT != 0;
    constexpr int kNumBufferWarps = kFwdDirect ? kNumScaleupWarps : kNumDataWarps;
    EP_STATIC_ASSERT(not kSmallBatch or kNumProxyWarps == kNumForwardWarps or (EP_SB_PROXY_MODE == 1 and kNumProxyWarps >= 2),
                     "Fewer proxy-slot warps than channels needs proxy mode 1");
    if constexpr (kSmallBatch)
        EP_DEVICE_ASSERT(blockDim.x == kNumThreads);

    // TMA buffers — one per DATA warp only (scale-up + forward).
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffer_layout = layout::BufferLayout<true>(token_layout, kNumBufferWarps, 1, smem);
    // The proxy warp never touches `tma_buffer`, clamp its index to 0.
    const auto tma_buffer = tma_buffer_layout
        .get_rank_buffer(warp_idx < kNumBufferWarps ? warp_idx : 0).get_token_buffer(0);

    // Proxy warp hand-off rings — placed in `smem[]` right after the TMA buffers.
    const auto proxy_ring_layout = ProxyRingLayout(
        kNumForwardWarps, kProxyRingDepth, tma_buffer_layout.get_buffer_end_ptr());
    for (int f = thread_idx; f < kNumForwardWarps; f += kNumThreads) {
        *proxy_ring_layout.get_head(f) = 0;
        *proxy_ring_layout.get_tail(f) = 0;
        *proxy_ring_layout.get_done(f) = 0;
    }
    // Tail publisher hand-off (small batch): per scale-up warp a sequence, a done flag and its
    // per-peer counts
    EP_STATIC_ASSERT(not kSmallBatch or EP_SB_PUT_SPLIT == 1 or EP_SB_PROXY_MODE == 1, "The put split needs proxy mode 1");
    constexpr bool kTailPublisher = kSmallBatch and EP_SB_TAIL_PUBLISHER != 0 and EP_SB_PROXY_MODE == 1 and
                                    kNumProxyWarps >= 2;
    constexpr int kTailPublisherWarpIdx = kNumDataWarps + 1;
    constexpr bool kRowTails = kTailPublisher and kRemoteFirst and EP_SB_SU_ROW_TAILS != 0;
    // Token-sized scale-out pieces (proxy mode 1): every piece is one token's bytes divided by
    // `kPiecesPerToken`, rounded up to 16 bytes
    constexpr int kPiecesPerToken = (kSmallBatch and EP_SB_PROXY_MODE == 1) ? EP_SB_PIECES_PER_TOKEN : 0;
    const auto sb_piece_bytes = [](const int& token_bytes) {
        return (math::ceil_div(token_bytes, kPiecesPerToken > 0 ? kPiecesPerToken : 1) + 15) / 16 * 16;
    };
    int* const pub_seq = reinterpret_cast<int*>(proxy_ring_layout.get_end_ptr());
    int* const pub_done = pub_seq + kNumScaleupWarps;
    int* const pub_counts = pub_done + kNumScaleupWarps;   // [kNumScaleupWarps][kNumScaleupRanks]
    if constexpr (kTailPublisher) {
        EP_STATIC_ASSERT(kNumScaleupWarps <= 32, "One publisher lane per scale-up warp");
        uint32_t num_dyn_smem;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(num_dyn_smem));
        EP_DEVICE_ASSERT(reinterpret_cast<int8_t*>(pub_counts + kNumScaleupWarps * kNumScaleupRanks) <= smem + num_dyn_smem);
        for (int i = thread_idx; i < kNumScaleupWarps * (2 + kNumScaleupRanks); i += kNumThreads)
            pub_seq[i] = 0;
    }
    // Small batch, compact scale-up walk: per scale-up warp and peer, the peer's linked list
    // reordered as remote-source entries first (token index, source token, source metadata),
    // placed after the tail publisher's hand-off. The host reserves the space
    // (`get_small_batch_scaleup_warps_per_channel`)
    constexpr int kSbListLen = kNumScaleoutRanks * kNumMaxTokensPerChannel + 1;
    constexpr bool kCompactWalk = kRemoteFirst and EP_SB_SU_COMPACT != 0 and 32 % kNumScaleupRanks == 0;
    // Expanded layout: the compact walk also loads a whole round's top-k slots at once
    constexpr int kRoundPeersPerHalf = (kNumTopk <= 32 and 32 % kNumTopk == 0) ? 32 / kNumTopk : 1;
    constexpr bool kRoundSlots = kCompactWalk and kUseExpandedLayout and EP_SB_SU_ROUND_SLOTS != 0 and
                                 kNumTopk <= 32 and 32 % kNumTopk == 0 and kNumScaleupRanks <= 2 * kRoundPeersPerHalf;
    int* const sb_list_base = pub_counts + kNumScaleupWarps * kNumScaleupRanks;
    if constexpr (kCompactWalk) {
        uint32_t num_dyn_smem;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(num_dyn_smem));
        EP_DEVICE_ASSERT(reinterpret_cast<int8_t*>(sb_list_base + kNumScaleupWarps * 3 * kNumScaleupRanks * kSbListLen) <=
                         smem + num_dyn_smem);
    }
#ifdef EP_SB_PROBE
    if (thread_idx == 0) {
        #pragma unroll
        for (int q = 1; q < 28; ++ q)
            sbc_probe[sm_idx][q] = 0ull;
        sbc_probe[sm_idx][0] = ptx::globaltimer();
    }
#endif
    __syncthreads();

    // All the buffer layouts
    const auto scaleup_buffer_root = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleupLayout, kNumScaleoutRanks * kNumMaxTokensPerRank,
        buffer);
    auto scaleup_buffer = scaleup_buffer_root;
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks,
        kNumChannels * (kNumMaxTokensPerChannel * (kAllowMultipleReduction ? 1 : kNumTopk)),
        scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks,
        kNumChannels * (kNumMaxTokensPerChannel * (kAllowMultipleReduction ? 1 : kNumTopk)),
        scaleout_recv_buffer.get_buffer_end_ptr());

    // Init TMA for scale-up and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if ((not kSmallBatch or warp_idx < kNumBufferWarps) and ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // NCCL Gin handle
    // For data warp, each warp is a channel, for proxy warp each lane is a channel.
    const auto gin_channel_in_sm = warp_idx < kNumDataWarps ?
        (warp_idx % kNumChannelsPerSM) :
        (kSmallBatch ? (warp_idx - kNumDataWarps) : (lane_idx < kNumForwardWarps ? lane_idx : 0));
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, gin_channel_in_sm);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    if constexpr (kSmallBatch) {
        // Barrier-free entry (see the small-batch mode notes above the template); the ring
        // initialisation above is already ordered by the block sync.
    } else {
        // Global parallel barriers for scale-out subteam and scale-up subteam
        // NOTES: this barrier needs a grid sync, as there are channel scale-up tail cleaning before
        comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                          kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
            gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    // The register moves are per warpgroup, so both roles must fill whole warpgroups
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and
                                  kNumScaleupWarps % 4 == 0 and not kUseExpandedLayout;
    constexpr int kNumRegistersForScaleupWarps = 40;
    constexpr int kNumRegistersForWarpPair = (kNumChannelsPerSM == 8) ? 192 : 256;
    constexpr int kNumRegistersForForwardWarps = (kSmallBatch and kNumChannelsPerSM == 4) ?
        EP_SB_FWD_REGS : kNumRegistersForWarpPair - kNumRegistersForScaleupWarps;
    constexpr bool kAdjustRegistersEx = kSmallBatch and EP_SB_EX_REGS != 0 and kUseExpandedLayout and
                                        kNumChannelsPerSM == 4 and kNumScaleupWarps % 4 == 0 and
                                        kNumDataWarps % 4 == 0 and kNumProxyWarps == 4;
    constexpr int kFwdExpectedTopk = (kSmallBatch and EP_SB_FWD_HOIST > 0) ?
        EP_SB_FWD_HOIST : math::constexpr_ceil_div(kNumTopk, kNumScaleoutRanks);


    // Scale-up send: push this channel's expert outputs (partials) to the scale-up peers in
    // `dst_mask` over NVLink, run by the scale-up warp of each channel
    EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up ranks for 64-bit mask");
    using scaleup_mask_t = std::conditional_t<(kNumScaleupRanks <= 32), uint32_t, uint64_t>;
    const auto send_partials = [&](const int& channel_idx, const scaleup_mask_t& dst_mask) {
        // Shift into the right buffer if using rank layout
        auto scaleup_buffer = [&]() {
            if constexpr (kUseScaleupRankLayout)
                return scaleup_buffer_root.get_rank_buffer(scaleup_rank_idx);
            else
                return scaleup_buffer_root;
        }();

        // Expanding send mode must not be backward
        if constexpr (kDoExpandedSend)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // Tail issuer
        // `st.release.sys` is pretty slow, so do it by an interval
        int update_counter = 0;
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        const auto tail_ptr = scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        constexpr bool kDeferTails = kSmallBatch and EP_SB_DEFER_TAILS != 0;
        bool tails_pending = false;
        const auto publish_tails = [&]() {
            ptx::tma_store_wait();
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                    if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                        {
                            if constexpr (kSmallBatch and EP_SB_RELAXED_TAILS != 0)
                                ptx::st_relaxed_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                            else
                                ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        }
                    stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                }
            }
            __syncwarp();
        };
        const auto publish_pending_tails = [&]() {
            if constexpr (kDeferTails) {
                if (tails_pending) {
                    publish_tails();
                    tails_pending = false;
                }
            }
        };
        int pub_seq_value = 0;
        // Row tails: a row was pushed since the last hand-off to the publisher
        bool sb_rows_pending = false;
        const auto update_tails = [&](const bool& finish = false) {
            if constexpr (kTailPublisher) {
                // Nothing new to publish: a hand-off would only cost the publisher a fence
                if constexpr (kRowTails) {
                    if (not finish and not sb_rows_pending)
                        return;
                    sb_rows_pending = false;
                }
                // Own bulk stores complete, then hand the counts to the publisher warp
                ptx::tma_store_wait();
                __syncwarp();
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    if (const auto j = i * 32 + lane_idx; j < kNumScaleupRanks)
                        pub_counts[warp_idx * kNumScaleupRanks + j] = stored_num_tokens_sent[i];
                }
                __syncwarp();
                ++ pub_seq_value;
                if (lane_idx == 0) {
                    ptx::st_release_cta(pub_seq + warp_idx, pub_seq_value);
                    if (finish)
                        ptx::st_release_cta(pub_done + warp_idx, 1);
                }
                __syncwarp();
                return;
            }
            if constexpr (kDeferTails) {
                if (finish) {
                    publish_tails();
                    tails_pending = false;
                } else {
                    tails_pending = true;
                }
                return;
            }
            ++ update_counter;
            if (finish or update_counter == kNumScaleupUpdateInterval) {
                // Wait all TMA stores to finish
                ptx::tma_store_wait();
                __syncwarp();

                // Issue
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                            {
                            if constexpr (kSmallBatch and EP_SB_RELAXED_TAILS != 0)
                                ptx::st_relaxed_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                            else
                                ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        }
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                update_counter = 0;
            }
            __syncwarp();
        };
        // Row tails, deferred form: hand the previous rows to the publisher once the next row's
        // load is in flight, so waiting for their bulk stores overlaps that load
        bool sb_remote_pass = false;
        const auto handoff_deferred_rows = [&]() {
            if constexpr (kRowTails and EP_SB_SU_ROW_TAILS == 2) {
                if (sb_remote_pass and sb_rows_pending)
                    update_tails();
            }
        };

        // Shape of `channel_linked_list`: `[kNumChannels, kNumMaxTokensPerChannel + 1, kNumScaleupRanks]`
        // Iterate until all scale-up peers finish
        constexpr int kForwardListTop = kNumScaleoutRanks * kNumMaxTokensPerChannel;
        int dst_scaleup_rank_idx = channel_idx;
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_ll_step[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_token_idx[i] = -1, stored_ll_step[i] = 1;
        constexpr int kMetadataStride = 2 + kNumTopk;
        int prefetched_src_global_token_idx[kNumScaleupRanksPerLane] = {};
        int prefetched_src_metadata_1[kNumScaleupRanksPerLane] = {};

        // Compact walk: reorder every owned peer's list into its remote-source entries (front,
        // list order) and its local-source entries (from the back, so slot `kSbListLen - 1 - r`
        // holds the r-th one). Lane `sub * kNumScaleupRanks + peer` loads entry `base + sub` of
        // that peer, so the whole list costs two load latencies instead of two per step
        int cw_num_remote = 0, cw_num_local = 0;
        int* const cw_tok = sb_list_base + warp_idx * (3 * kNumScaleupRanks * kSbListLen);
        int* const cw_src = cw_tok + kNumScaleupRanks * kSbListLen;
        int* const cw_meta1 = cw_src + kNumScaleupRanks * kSbListLen;
        if constexpr (kCompactWalk) {
            constexpr int kLanesPerPeer = 32 / kNumScaleupRanks;
            const int peer = lane_idx % kNumScaleupRanks, sub = lane_idx / kNumScaleupRanks;
            // Bit `q` of the result: lane `q * kNumScaleupRanks + peer` of the ballot
            const auto peer_bits = [&](const uint32_t& ballot) {
                uint32_t bits = 0;
                #pragma unroll
                for (int q = 0; q < kLanesPerPeer; ++ q)
                    bits |= ((ballot >> (q * kNumScaleupRanks + peer)) & 1u) << q;
                return bits;
            };
            const uint32_t below = (1u << sub) - 1u;
            const bool owned_peer = (dst_mask >> peer) & 1;
            // All list entries and their metadata are loaded before any is classified (two load
            // latencies in total). Entries past a list's end are stale (the list is not cleared),
            // so their metadata loads are bounded by the received token count and their values
            // are dropped below
            // The list holds two FIFOs (see the dispatch forwarder): remote-source entries from 0
            // upward, local-source ones from kForwardListTop downward, each ended by -1
            constexpr int kNumChunks0 = math::constexpr_ceil_div((kNumScaleoutRanks - 1) * kNumMaxTokensPerChannel + 1, kLanesPerPeer);
            constexpr int kNumChunks1 = math::constexpr_ceil_div(kNumMaxTokensPerChannel + 1, kLanesPerPeer);
            constexpr int kNumChunks = kNumChunks0 + kNumChunks1;
            const auto chunk_entry = [&](const int& c) {
                return c < kNumChunks0 ? c * kLanesPerPeer + sub : kForwardListTop - ((c - kNumChunks0) * kLanesPerPeer + sub);
            };
            int chunk_tok[kNumChunks], chunk_src[kNumChunks], chunk_meta1[kNumChunks];
            #pragma unroll
            for (int c = 0; c < kNumChunks; ++ c) {
                const int s = chunk_entry(c);
                chunk_tok[c] = (owned_peer and 0 <= s and s < kSbListLen) ?
                    __ldg(channel_linked_list + (channel_idx * kSbListLen + s) * kNumScaleupRanks + peer) : -1;
            }
            #pragma unroll
            for (int c = 0; c < kNumChunks; ++ c) {
                const bool in_bounds = chunk_tok[c] >= 0 and chunk_tok[c] < num_reduced_tokens;
                chunk_src[c] = in_bounds ? __ldg(src_metadata + chunk_tok[c] * kMetadataStride + 0) : -1;
                chunk_meta1[c] = in_bounds ? __ldg(src_metadata + chunk_tok[c] * kMetadataStride + 1) : -1;
            }
            bool ended = not owned_peer;
            #pragma unroll
            for (int c = 0; c < kNumChunks; ++ c) {
                if (c == kNumChunks0)
                    ended = not owned_peer;
                const int tok = ended ? -1 : chunk_tok[c];
                // Entries after a list's terminator are stale
                const uint32_t neg = peer_bits(__ballot_sync(0xffffffffu, tok < 0));
                const bool valid = tok >= 0 and (neg & below) == 0;
                ended = ended or neg != 0;
                const int src = chunk_src[c], meta1 = chunk_meta1[c];
                const bool is_remote = valid and src / (kNumMaxTokensPerRank * kNumScaleupRanks) != scaleout_rank_idx;
                const uint32_t remote_bits = peer_bits(__ballot_sync(0xffffffffu, is_remote));
                const uint32_t local_bits = peer_bits(__ballot_sync(0xffffffffu, valid and not is_remote));
                if (valid) {
                    const int pos = is_remote ? cw_num_remote + __popc(remote_bits & below) :
                                                kSbListLen - 1 - (cw_num_local + __popc(local_bits & below));
                    cw_tok[peer * kSbListLen + pos] = tok;
                    cw_src[peer * kSbListLen + pos] = src;
                    cw_meta1[peer * kSbListLen + pos] = meta1;
                }
                cw_num_remote += __popc(remote_bits);
                cw_num_local += __popc(local_bits);
            }
            __syncwarp();
        }

        // With `kRemoteFirst`, pass 0 sends the partials of remote-source tokens and pass 1 the
        // local-source ones; the forwarders consume them in the same order
        for (int pass = 0; pass < kNumOrderPasses; ++ pass) {
        int cw_round = 0;
        sb_remote_pass = kRemoteFirst and pass == 0;
        // Pass 0 walks the remote FIFO up from 0, pass 1 the local FIFO down from the top
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_ll_idx[i] = sb_remote_pass or not kRemoteFirst ? 0 : kForwardListTop,
            stored_ll_step[i] = sb_remote_pass or not kRemoteFirst ? 1 : -1;
        while (true) {
            // Load token indices in the list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                const auto j = i * 32 + lane_idx;
                const bool owned = (i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) and
                                   ((dst_mask >> (j % kNumScaleupRanks)) & 1);
                if constexpr (kCompactWalk) {
                    const bool has = owned and cw_round < (pass == 0 ? cw_num_remote : cw_num_local);
                    const int pos = j * kSbListLen + (pass == 0 ? cw_round : kSbListLen - 1 - cw_round);
                    stored_token_idx[i] = has ? cw_tok[pos] : -1;
                    prefetched_src_global_token_idx[i] = has ? cw_src[pos] : -1;
                    prefetched_src_metadata_1[i] = has ? cw_meta1[pos] : -1;
                    continue;
                }
                const auto list = channel_linked_list + channel_idx * (kForwardListTop + 1) * kNumScaleupRanks + j;
                stored_token_idx[i] = owned ? __ldg(list + stored_ll_idx[i] * kNumScaleupRanks) : -1;
                // The first FIFO ended: continue with the second one, from the top down (the
                // remote-first walk visits it in its own pass instead)
                if (not kRemoteFirst and stored_token_idx[i] < 0 and stored_ll_step[i] > 0) {
                    stored_ll_idx[i] = kForwardListTop, stored_ll_step[i] = -1;
                    stored_token_idx[i] = owned ? __ldg(list + stored_ll_idx[i] * kNumScaleupRanks) : -1;
                }
                // Small batch: fetch this step's source metadata lane-parallel, one latency for
                // all peers instead of one per token inside the loop below
                if constexpr (kSmallBatch) {
                    prefetched_src_global_token_idx[i] = stored_token_idx[i] >= 0 ?
                        __ldg(src_metadata + stored_token_idx[i] * kMetadataStride + 0) : -1;
                    prefetched_src_metadata_1[i] = stored_token_idx[i] >= 0 ?
                        __ldg(src_metadata + stored_token_idx[i] * kMetadataStride + 1) : -1;
                }
                if constexpr (kSmallBatch and EP_SB_SU_PREFETCH != 0 and not kUseExpandedLayout) {
                    bool in_pass = stored_token_idx[i] >= 0;
                    if constexpr (kRemoteFirst) {
                        const int src_node = prefetched_src_global_token_idx[i] / (kNumMaxTokensPerRank * kNumScaleupRanks);
                        in_pass = in_pass and ((src_node != scaleout_rank_idx) == (pass == 0));
                    }
                    if (in_pass)
                        ptx::prefetch_l2_bulk(math::advance_ptr(x, static_cast<int64_t>(stored_token_idx[i]) * kNumHiddenBytes), kNumHiddenBytes);
                }
            }
            __syncwarp();

            // Check whether all ranks are finished
            bool exited = true;
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                exited &= ptx::all(stored_token_idx[i] < 0);
            if (exited)
                break;

            // Expanded layout with the compact walk: load the top-k slots of every row of this
            // round together, one load latency per round instead of one per row. Lane L holds
            // slot `L % kNumTopk` of the rows of peers `L / kNumTopk` and that plus
            // `kRoundPeersPerHalf`
            int round_slots[2] = {-1, -1};
            if constexpr (kRoundSlots) {
                #pragma unroll
                for (int h = 0; h < 2; ++ h) {
                    const int p = lane_idx / kNumTopk + h * kRoundPeersPerHalf;
                    const int tok = __shfl_sync(0xffffffffu, stored_token_idx[0], p);
                    if (p < kNumScaleupRanks and tok >= 0)
                        round_slots[h] = __ldg(src_metadata + tok * kMetadataStride + 2 + lane_idx % kNumTopk);
                }
            }

            // Process tokens for all ranks together using bitmask to skip inactive ranks
            using mask_t = scaleup_mask_t;
            mask_t wip_mask = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                bool in_pass = stored_token_idx[j] >= 0;
                if constexpr (kRemoteFirst) {
                    const int src_node = prefetched_src_global_token_idx[j] / (kNumMaxTokensPerRank * kNumScaleupRanks);
                    in_pass = in_pass and ((src_node != scaleout_rank_idx) == (pass == 0));
                }
                wip_mask |= static_cast<mask_t>(ptx::gather(in_pass)) << (j * 32);
            }
            while (wip_mask) {
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)
                const auto start = (dst_scaleup_rank_idx + 1) % kNumScaleupRanks;
                const auto hi_mask = (wip_mask >> start) << start;
                dst_scaleup_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);
                wip_mask ^= static_cast<mask_t>(1) << dst_scaleup_rank_idx;

                // Exchange token index (and prefetched metadata) from the owning lane using static partition iteration
                int token_idx = -1, pf_src_global_token_idx = -1, pf_src_metadata_1 = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;
                    if constexpr (kSmallBatch) {
                        pf_src_global_token_idx = src_lane_idx == lane_idx ? prefetched_src_global_token_idx[j] : pf_src_global_token_idx;
                        pf_src_metadata_1 = src_lane_idx == lane_idx ? prefetched_src_metadata_1[j] : pf_src_metadata_1;
                    }
                }
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);

                // Get source metadata and decide the destination buffer
                const auto src_global_token_idx = kSmallBatch ?
                    ptx::exchange(pf_src_global_token_idx, dst_scaleup_rank_idx % 32) :
                    __ldg(src_metadata + token_idx * kMetadataStride + 0);
                const auto src_metadata_1 = kSmallBatch ?
                    ptx::exchange(pf_src_metadata_1, dst_scaleup_rank_idx % 32) :
                    __ldg(src_metadata + token_idx * kMetadataStride + 1);
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * kNumScaleupRanks);
                auto token_buffer = [&]() {
                    if constexpr (kUseScaleupRankLayout) {
                        const auto src_slot_idx = src_metadata_1 / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        const auto master_topk_idx = src_metadata_1 % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                    }
                }();
                token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));
                if constexpr (kSmallBatch and EP_SB_DIAG_NO_SU != 0) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                        stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                    __syncwarp();
                    continue;
                }

                // Some checks
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                int stored_topk_slot_idx = -1;
                if constexpr (kRoundSlots) {
                    const int p = dst_scaleup_rank_idx;
                    const int src_lane = (p % kRoundPeersPerHalf) * kNumTopk + lane_idx % kNumTopk;
                    const int v0 = __shfl_sync(0xffffffffu, round_slots[0], src_lane);
                    const int v1 = __shfl_sync(0xffffffffu, round_slots[1], src_lane);
                    stored_topk_slot_idx = lane_idx < kNumTopk ? (p < kRoundPeersPerHalf ? v0 : v1) : -1;
                } else if constexpr (kUseExpandedLayout) {
                    if (lane_idx < kNumTopk)
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    __syncwarp();
                }

                // 3 cases:
                //  - no-expand, expand + no-reduce
                //  - expand + reduce
                //  - expand + send all
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    int token_idx_in_tensor = token_idx;
                    if constexpr (kUseExpandedLayout)
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // Directly load
                    if (ptx::elect_one_sync()) {
                        const auto load_ptr =
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        // Reusing the staging buffer only needs the previous store to have read it
                        if constexpr (kSmallBatch) ptx::tma_store_wait_read(); else ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();
                    publish_pending_tails();
                    handoff_deferred_rows();
                } else if constexpr (kAllowMultipleReduction) {
                    publish_pending_tails();
                    // Do local reduction
                    // Sort valid top-k indices to front
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kSmallBatch ? EP_SB_SU_UNROLL : 4>();
                    constexpr int kSuExpectedTopk = (kSmallBatch and EP_SB_SU_HOIST > 0) ?
                        EP_SB_SU_HOIST : math::constexpr_ceil_div(kNumTopk, kNumRanks);
                    combine_reduce<kHiddenVec, kUnrollFactor, kSuExpectedTopk>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [&]() {
                            if constexpr (kSmallBatch) ptx::tma_store_wait_read(); else ptx::tma_store_wait();
                            __syncwarp();
                            handoff_deferred_rows();
                        }
                    );
                    ptx::tma_store_fence();
                    __syncwarp();
                } else {
                    publish_pending_tails();
                    // No local reduction, send all data (expanded send)
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (topk_slot_idx < 0)
                            continue;

                        if (ptx::elect_one_sync()) {
                            // Load
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);
                            if constexpr (kSmallBatch) ptx::tma_store_wait_read(); else ptx::tma_store_wait();
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode

                            // Store
                            const auto dst_token_buffer = scaleup_buffer
                                .get_rank_buffer(k)
                                .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                            ptx::tma_store_1d(
                                gin.get_sym_ptr<ncclTeamTagLsa>(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx),
                                tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            ptx::tma_store_commit();
                        }
                        __syncwarp();
                    }
                }

                // Write top-k weights (expanded send handled inside the loop above)
                if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
                    float value = 0;
                    if constexpr (kUseExpandedLayout) {
                        if (stored_topk_slot_idx >= 0)
                            value = __ldg(topk_weights + stored_topk_slot_idx);
                    } else {
                        value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));
                    }
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Issue TMA stores into remote scale-up buffer
                // NOTES: `kDoExpandedSend` mode has already issued
                if (not kDoExpandedSend and ptx::elect_one_sync()) {
                    // Wait TMA arrival (only for non-reduced cases)
                    if (no_local_reduce) {
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
                // Remote pass: hand every row to the publisher as soon as it is stored (mode 1),
                // or once the next row's load is in flight (mode 2), so the forwarder of that
                // peer does not wait for the rest of the round
                if constexpr (kRowTails) {
                    sb_rows_pending = true;
                    if (EP_SB_SU_ROW_TAILS == 1 and pass == 0)
                        update_tails();
                }
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            update_tails();

            // Move linked list
            if constexpr (kCompactWalk) {
                ++ cw_round;
            } else {
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                    stored_ll_idx[i] += stored_token_idx[i] >= 0 ? stored_ll_step[i] : 0;
            }
        }
        // The remote pass's partials gate the scale-out puts: publish them before the local pass
        // (the publisher hands over every step anyway, and `finish` would mark the warp done)
        if (kRemoteFirst and pass == 0) {
            update_tails(not kTailPublisher);
            // The SM's push rate is the limit, so no warp starts its local rows before every
            // warp of the SM has pushed its remote ones
            if constexpr (kSmallBatch and EP_SB_SU_PASS_BARRIER != 0)
                asm volatile("bar.sync 1, %0;" :: "r"(kNumScaleupWarps * 32) : "memory");
#ifdef EP_SB_PROBE
            {
                int sent0 = 0;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    sent0 += stored_num_tokens_sent[j];
                sent0 = ptx::reduce_add(sent0);
                if (lane_idx == 0) {
                    const auto t = ptx::globaltimer();
                    atomicMax(&sbc_probe[sm_idx][25], t);
                    sbc_su[sm_idx * 32 + warp_idx][0] = static_cast<unsigned long long>(sent0);
                    sbc_su[sm_idx * 32 + warp_idx][1] = t;
                }
            }
#endif
        }
        }

        // Update for the unissued ones
        update_tails(true);
#ifdef EP_SB_PROBE
        {
            int sent = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                sent += stored_num_tokens_sent[j];
            sent = ptx::reduce_add(sent);
            if (lane_idx == 0) {
                atomicMax(&sbc_probe[sm_idx][1], ptx::globaltimer());
                atomicMax(&sbc_probe[sm_idx][7], static_cast<unsigned long long>(sent));
                sbc_su[sm_idx * 32 + warp_idx][2] = static_cast<unsigned long long>(sent);
                sbc_su[sm_idx * 32 + warp_idx][3] = ptx::globaltimer();
            }
        }
#endif
    };

    // Scale-up warps per channel: one normally; the small batch may run several, each sending
    // to the peers `p` with `p % kNumScaleupWarpsPerChannel == subset`. One warp moves about
    // 14 KB per 3 us (one bulk store to a peer at a time), so at decode sizes the send phase is
    // bounded by the longest per-warp chain and splitting a channel's peers over warps shortens it.
    EP_STATIC_ASSERT(kNumScaleupWarps % kNumChannelsPerSM == 0, "Scale-up warps must be a multiple of the channels");
    constexpr int kNumScaleupWarpsPerChannel = kNumScaleupWarps / kNumChannelsPerSM;
    EP_STATIC_ASSERT(kSmallBatch or kNumScaleupWarpsPerChannel == 1, "Only the small batch runs several scale-up warps per channel");
    const auto scaleup_subset_mask = [&](const int& subset) {
        scaleup_mask_t mask = 0;
        #pragma unroll
        for (int p = 0; p < kNumScaleupRanks; ++ p)
            mask |= static_cast<scaleup_mask_t>(p % kNumScaleupWarpsPerChannel == subset) << p;
        return mask;
    };

    // The proxy warpgroup hands registers to the forward warps (expanded layout only)
    if constexpr (kAdjustRegistersEx) {
        if (warp_idx >= kNumDataWarps)
            ptx::warpgroup_reg_dealloc<64>();
    }

    // Different warp roles
    if (warp_idx < kNumScaleupWarps) {
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx % kNumChannelsPerSM;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

        send_partials(channel_idx, scaleup_subset_mask(warp_idx / kNumChannelsPerSM));
    } else if (warp_idx < kNumDataWarps) {
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

        // Indexed-signal id owned by this channel within its GIN context. Uses the
        // companion helper to `get_qp_mode` (called at kernel entry) so the id is
        // unique among all channels sharing this warp's QP, regardless of which of
        // `get_qp_mode`'s branches picked the mapping.
        const auto signal_id = channel_signal_id(forward_warp_idx);

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();
        else if constexpr (kAdjustRegistersEx)
            ptx::warpgroup_reg_alloc<192>();

#ifdef EP_SB_PROBE
        if (lane_idx < 8)
            sbc_chan[channel_idx][lane_idx] = 0ull;
        __syncwarp();
#endif
        if constexpr (kSmallBatch and EP_SB_L2_PREFETCH != 0 and not kUseExpandedLayout) {
            // Rows of `x` (one per received token in this layout), strided over all forward lanes
            EP_STATIC_ASSERT(kNumHiddenBytes % 16 == 0, "Unaligned hidden for the bulk prefetch");
            constexpr int kNumPrefetchLanes = kNumSMs * kNumForwardWarps * 32;
            for (int row = (sm_idx * kNumForwardWarps + forward_warp_idx) * 32 + lane_idx;
                 row < num_reduced_tokens; row += kNumPrefetchLanes)
                ptx::prefetch_l2_bulk(math::advance_ptr(x, static_cast<int64_t>(row) * kNumHiddenBytes), kNumHiddenBytes);
        }

        constexpr int kNumSlotsPerChannel = kNumMaxTokensPerChannel * (kAllowMultipleReduction ? 1 : kNumTopk);
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer<kNumSlotsPerChannel>(channel_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumSlotsPerChannel>(channel_idx);

        int slot_of_per_channel[kNumScaleoutRanks] = {};

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // Per Channel/ScaleOutRank batch metadata
        // Slots this channel owes each scale-out peer, counted DOWN as they are recorded;
        // reaching 0 means that destination is finished.
        int num_slots_remaining_per_dst[kNumScaleoutRanks] = {};
        count_forward_slots_per_dst<kAllowMultipleReduction, kUseExpandedLayout, kNumScaleoutRanks,
                                    kNumScaleupRanks, kNumMaxTokensPerRank, kNumMaxTokensPerChannel,
                                    kNumTopk, kNumForwardMetadataDims>(
            token_metadata_at_forward, lane_idx, num_slots_remaining_per_dst);
        __syncwarp();
        // Slots this channel still owes remote peers in total; at 0 the ring can be retired.
        int num_remote_slots_remaining = 0;
        #pragma unroll
        for (int d = 0; d < kNumScaleoutRanks; ++ d)
            if (d != scaleout_rank_idx)
                num_remote_slots_remaining += num_slots_remaining_per_dst[d];

        int batch_count[kNumScaleoutRanks] = {};
        int batch_start_slot[kNumScaleoutRanks];

        ProxyPutDesc* const my_ring = proxy_ring_layout.get_ring(forward_warp_idx);
        unsigned& my_head = *proxy_ring_layout.get_head(forward_warp_idx);
        const unsigned* const my_tail_ptr = proxy_ring_layout.get_tail(forward_warp_idx);

        // Hand a completed batch to the proxy instead of issuing the put here. Callers
        // must be under `elect_one_sync`. Backpressure: spin while ring is full.
        const auto issue_batched_rdma = [&](const int& dst) {
            if (batch_count[dst] == 0)
                return;
            // Small batch: the per-token waits below only cover the smem read, so the puts
            // need the stores fully completed here
            if constexpr (kSmallBatch)
                ptx::tma_store_wait();
            const auto send_ptr = scaleout_send_buffer
                .get_rank_buffer(dst)
                .get_token_buffer(batch_start_slot[dst])
                .get_base_ptr();
            const auto recv_ptr = scaleout_recv_buffer
                .get_rank_buffer(scaleout_rank_idx)
                .get_token_buffer(batch_start_slot[dst])
                .get_base_ptr();

            // Wait for a free ring slot. Timed: this is the producer half of the hand-off with
            // the proxy warp, so if the proxy stops draining this ring the warp parks here with
            // nothing to report — the SM stays at 100% and no timeout or Xid fires.
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                const auto ring_tail = ptx::ld_acquire_cta(my_tail_ptr);
                if (my_head - ring_tail < static_cast<unsigned>(kProxyRingDepth))
                    return true;

                if (is_last_check) {
                    printf("DeepEP combine (proxy ring full) timeout, scale-out: %d, scale-up: %d, "
                           "channel: %d, forward warp: %d, dst: %d, head: %u, tail: %u, depth: %d\n",
                           scaleout_rank_idx, scaleup_rank_idx, channel_idx, forward_warp_idx,
                           dst, my_head, ring_tail, kProxyRingDepth);
                }
                return false;
            });

            const unsigned slot = my_head % static_cast<unsigned>(kProxyRingDepth);
            my_ring[slot].send_ptr = send_ptr;
            my_ring[slot].recv_ptr = recv_ptr;
            my_ring[slot].num_bytes = batch_count[dst] * static_cast<int>(token_layout.get_num_bytes<false>());
            if constexpr (kSmallBatch and EP_SB_DIAG_PUT_BYTES > 0)
                my_ring[slot].num_bytes = min(my_ring[slot].num_bytes, EP_SB_DIAG_PUT_BYTES);
            my_ring[slot].dst = dst;

            // Small batch: the descriptor is CTA-local and the proxy's GIN post fences at
            // system scope before its doorbell, cumulative over what the proxy acquired from this
            // release, so the device-scope fence can go (`EP_SB_HANDOFF_FENCE=0`)
            if constexpr (not kSmallBatch or EP_SB_HANDOFF_FENCE != 0)
                __threadfence();
            ptx::st_release_cta(&my_head, my_head + 1);

            batch_count[dst] = 0;
        };

        // Update per Channel/ScaleOutRank batch metadata and GIN PUT when threshold met
        const auto record_slot_and_maybe_flush = [&](const int& dst, const int& slot) {
            if (batch_count[dst] == 0)
                batch_start_slot[dst] = slot;
            batch_count[dst] += 1;

            // A destination whose tokens are all handled will get no more, so forward its batch
            // now instead of waiting for it to fill.
            const bool dst_exhausted = (-- num_slots_remaining_per_dst[dst] == 0);
            if (batch_count[dst] == kBatchSize or dst_exhausted)
                issue_batched_rdma(dst);

            // All tokens for every destination are used: tell the proxy immediately so it can
            // stop waiting for more descriptors on this ring.
            if (-- num_remote_slots_remaining == 0)
                ptx::st_release_cta(proxy_ring_layout.get_done(forward_warp_idx), 1);
        };

        int last_src_scaleout_rank_idx = -1;
        int last_slot_written = -1;
        const auto flush_last_tma_and_record_batch = [&]() {
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {
                if constexpr (kSmallBatch) ptx::tma_store_wait_read(); else ptx::tma_store_wait();

                 // Issue only if not local rank
                if (last_src_scaleout_rank_idx != scaleout_rank_idx)
                    record_slot_and_maybe_flush(last_src_scaleout_rank_idx, last_slot_written);
            }
            __syncwarp();
        };

        // Replay the dispatch. The forward list holds two FIFOs (see the dispatch forwarder):
        // remote-source tokens from entry 0 upward, local-source tokens from kForwardListTop
        // downward, each ended by -1. The high-throughput walk follows the first with the
        // second; the small batch walks one FIFO per pass (remote first), matching the
        // scale-up senders, and the order within one source keeps the per-source slot
        // numbering of the scale-out buffers
        constexpr int kForwardListTop = kNumScaleoutRanks * kNumMaxTokensPerChannel;
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        for (int pass = 0; pass < kNumOrderPasses; ++ pass) {
        for (int i = (kRemoteFirst and pass == 1) ? kForwardListTop : 0, step = (kRemoteFirst and pass == 1) ? -1 : 1; ; i += step) {
            auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            if (not kRemoteFirst and src_token_global_idx < 0 and step > 0) {
                i = kForwardListTop, step = -1;
                src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            }
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                break;
            if constexpr (kRemoteFirst)
                EP_DEVICE_ASSERT((src_scaleout_rank_idx != scaleout_rank_idx) == (pass == 0));

            // Scaleup rank mask
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up peers");
            using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
            const auto scaleup_mask = ptx::reduce_or(
                stored_src_scaleup_rank_idx >= 0 ?
                (mask_t(1) << stored_src_scaleup_rank_idx) : mask_t(0));
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_is_scaleup_rank_needed[j] = (scaleup_mask >> (j * 32 + lane_idx)) & 1;

            // Wait all tails to arrive
#ifdef EP_SB_PROBE
            const auto probe_wait_t0 = ptx::globaltimer();
#endif
            if constexpr (not (kSmallBatch and EP_SB_DIAG_NO_WAIT != 0))
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                bool arrived = true;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                if (ptx::all(arrived))
                    return true;

                // Reload cached
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto k = j * 32 + lane_idx;
                    stored_cached_scaleup_tail[j] = j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks ?
                        ptx::ld_acquire_sys(scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                               channel_idx, j * 32 + lane_idx,
                               stored_num_tokens_recv[j],
                               stored_cached_scaleup_tail[j],
                               stored_is_scaleup_rank_needed[j]);
                    }
                }
                return false;
            });

#ifdef EP_SB_PROBE
            const auto probe_reduce_t0 = ptx::globaltimer();
            if (lane_idx == 0) {
                if (i == 0)
                    atomicMax(&sbc_probe[sm_idx][2], probe_reduce_t0);
                atomicAdd(&sbc_probe[sm_idx][12], probe_reduce_t0 - probe_wait_t0);
                atomicAdd(&sbc_probe[sm_idx][15], 1ull);
                if (kRemoteFirst and pass == 0) {
                    atomicAdd(&sbc_probe[sm_idx][22], 1ull);
                    atomicAdd(&sbc_probe[sm_idx][23], probe_reduce_t0 - probe_wait_t0);
                    if (sbc_chan[channel_idx][0] == 0)
                        sbc_chan[channel_idx][6] = probe_reduce_t0;
                    sbc_chan[channel_idx][0] += 1;
                    sbc_chan[channel_idx][2] += probe_reduce_t0 - probe_wait_t0;
                } else {
                    sbc_chan[channel_idx][4] += 1;
                }
            }
#endif
            // Increase received count
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);
            
            if constexpr (not kAllowMultipleReduction) {
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                auto topk_valid_mask = kUseExpandedLayout ?
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                const int num_valid_topk = __popc(topk_valid_mask);
                if (ptx::elect_one_sync()) {
                    int packed_offset_in_entry = 0;
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        if ((topk_valid_mask & (1u << k)) == 0u)
                            continue;

                        // Issue TMA load, and wait
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        const int slot = slot_of_per_channel[src_scaleout_rank_idx] + packed_offset_in_entry;
                        const auto recv_buffer_ptr = scaleout_recv_buffer
                            .get_rank_buffer(scaleout_rank_idx)
                            .get_token_buffer(slot)
                            .get_base_ptr();
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr :
                            scaleout_send_buffer
                                .get_rank_buffer(src_scaleout_rank_idx)
                                .get_token_buffer(slot)
                                .get_base_ptr();
                        ++ packed_offset_in_entry;
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        if constexpr (kSmallBatch) ptx::tma_store_wait_read(); else ptx::tma_store_wait();

                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx)
                            record_slot_and_maybe_flush(src_scaleout_rank_idx, slot);
                    }
                }
                __syncwarp();
                slot_of_per_channel[src_scaleout_rank_idx] += num_valid_topk;
            } else {
                // NOTES: we must do deduplicate and only add once from one rank
                auto reduce_valid_mask = ptx::gather(
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // Calculate the source buffer index
                int stored_src_buffer_idx = 0;
                if constexpr (kUseScaleupRankLayout) {
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }
                
                // Small batch: the merged weights (read from the master lane's partial) are loaded
                // now so their latency overlaps the reduce
                float early_topk_weight = 0;
                if constexpr (kSmallBatch and EP_SB_EARLY_WEIGHTS != 0) {
                    const auto master_buffer_idx = ptx::exchange(
                        stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                    if (stored_src_scaleup_rank_idx >= 0)
                        early_topk_weight = scaleup_buffer.get_token_buffer(master_buffer_idx, true).get_topk_weights_ptr()[lane_idx];
                }

                // Preprocess top-k indices
                int topk_slot_idx[kNumTokensInScaleupLayout];
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,
                    [=](const int& idx) {
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                // Do reduce
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? (kSmallBatch ? EP_SB_FWD_UNROLL : 8) : (kSmallBatch ? EP_SB_FWD_UNROLL_EX : 4)>();
                if constexpr (kFwdDirect) {
                    // Reduce straight into the scale-out send slot (or the local receive slot). No
                    // staging buffer and no bulk store to wait for: the batch's release to the proxy
                    // orders these stores, and the proxy's GIN post fences at system scope before
                    // the NIC reads them
                    const int recv_slot = slot_of_per_channel[src_scaleout_rank_idx];
                    const auto recv_token_buffer = scaleout_recv_buffer
                        .get_rank_buffer(scaleout_rank_idx)
                        .get_token_buffer(recv_slot);
                    const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                        recv_token_buffer :
                        scaleout_send_buffer.get_rank_buffer(src_scaleout_rank_idx).get_token_buffer(recv_slot);
                    slot_of_per_channel[src_scaleout_rank_idx] += 1;
                    if constexpr (EP_SB_DIAG_NO_REDUCE == 0) {
                    combine_reduce<kHiddenVec, kUnrollFactor, kFwdExpectedTopk>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(send_token_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                        },
                        /* Wait buffer release */ []() {}
                    );
                    }
                    stored_src_buffer_idx = ptx::exchange(
                        stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                    if (stored_src_scaleup_rank_idx >= 0) {
                        send_token_buffer.get_topk_weights_ptr()[lane_idx] =
                            scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true).get_topk_weights_ptr()[lane_idx];
                    }
                    __syncwarp();
                    if (src_scaleout_rank_idx != scaleout_rank_idx and ptx::elect_one_sync())
                        record_slot_and_maybe_flush(src_scaleout_rank_idx, recv_slot);
                    __syncwarp();
                } else {
                if constexpr (kSmallBatch and EP_SB_DIAG_NO_REDUCE != 0)
                    flush_last_tma_and_record_batch();
                else
                combine_reduce<kHiddenVec, kUnrollFactor, kFwdExpectedTopk>(
                    lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    /* Get source base */ [=](const int& slot_idx) {
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ [=]() {
                        flush_last_tma_and_record_batch();
                    }
                );

                // Merge topk weights
                // NOTES: the slot indices must follow the master lane
                if constexpr (kSmallBatch and EP_SB_EARLY_WEIGHTS != 0) {
                    if (stored_src_scaleup_rank_idx >= 0)
                        tma_buffer.get_topk_weights_ptr()[lane_idx] = early_topk_weight;
                } else {
                    stored_src_buffer_idx = ptx::exchange(
                        stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                    if (stored_src_scaleup_rank_idx >= 0) {
                        tma_buffer.get_topk_weights_ptr()[lane_idx] =
                            scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                        .get_topk_weights_ptr()[lane_idx];
                    }
                }
                ptx::tma_store_fence();
                __syncwarp(); // Necessary to let the leader lane see the writes

                // Assign send and receive buffers
                // NOTES: as we only have 1 destination, we will use "send" as "recv" for local transfer
                const int recv_slot = slot_of_per_channel[src_scaleout_rank_idx];
                const auto recv_token_buffer = scaleout_recv_buffer
                    .get_rank_buffer(scaleout_rank_idx)
                    .get_token_buffer(recv_slot);
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                    recv_token_buffer :
                    scaleout_send_buffer
                    .get_rank_buffer(src_scaleout_rank_idx)
                    .get_token_buffer(slot_of_per_channel[src_scaleout_rank_idx]);
                slot_of_per_channel[src_scaleout_rank_idx] += 1;

                // Write into scale-out send buffer or local rank recv buffer bypass
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // Record RDMA info to issue later
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_slot_written = recv_slot;
                }
#ifdef EP_SB_PROBE
                if (lane_idx == 0)
                    atomicAdd(&sbc_probe[sm_idx][13], ptx::globaltimer() - probe_reduce_t0);
                if (lane_idx == 0 and kRemoteFirst and pass == 0) {
                    atomicAdd(&sbc_probe[sm_idx][24], ptx::globaltimer() - probe_reduce_t0);
                    sbc_chan[channel_idx][3] += ptx::globaltimer() - probe_reduce_t0;
                }
#endif
            }
        }
        // End of the remote pass: hand the last remote token and the partial batches to the
        // proxy now instead of after the local pass
        if constexpr (kRemoteFirst) {
            if (pass == 0) {
                if constexpr (kAllowMultipleReduction) {
                    flush_last_tma_and_record_batch();
                    last_src_scaleout_rank_idx = -1;
                }
                if (ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int dst = 0; dst < kNumScaleoutRanks; ++ dst) {
                        if (dst != scaleout_rank_idx)
                            issue_batched_rdma(dst);
                    }
                    if constexpr (EP_SB_EARLY_DONE != 0)
                        ptx::st_release_cta(proxy_ring_layout.get_done(forward_warp_idx), 1);
#ifdef EP_SB_PROBE
                    const auto t = ptx::globaltimer();
                    atomicMax(&sbc_probe[sm_idx][19], t);
                    atomicMax(&sbc_probe[sm_idx][21], ~0ull - t);
                    sbc_chan[channel_idx][1] = t;
#endif
                }
                __syncwarp();
            }
        }
        }

        // Issue the last TMA and record operation for last RDMA
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_record_batch();

        // Issue the last RDMA per channel/scaleout-warp
        if (ptx::elect_one_sync()) {
            // Small batch: complete every outstanding store (the per-token waits only covered
            // the smem reads), including the local-bypass ones the reduce epilogue reads
            if constexpr (kSmallBatch)
                ptx::tma_store_wait();
            #pragma unroll
            for (int dst = 0; dst < kNumScaleoutRanks; ++ dst) {
                if constexpr (kRemoteFirst and EP_SB_EARLY_DONE != 0)
                    EP_DEVICE_ASSERT(batch_count[dst] == 0);
                issue_batched_rdma(dst);
            }

            ptx::st_release_cta(proxy_ring_layout.get_done(forward_warp_idx), 1);
#ifdef EP_SB_PROBE
            atomicMax(&sbc_probe[sm_idx][3], ptx::globaltimer());
#endif
        }
        __syncwarp();

        // Clean scaleup tails
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const auto k = j * 32 + lane_idx;
            if (j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks)
                *scaleup_tail_ptr(channel_idx, k) = 0;
        }
        __syncwarp();

        // Small batch: the forwarders wait for their own channel's inbound puts (the
        // high-throughput proxy waits for the whole SM instead)
        if constexpr (kSmallBatch) {
        // Update, wait and clean
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");

        // Derive the expected inbound put count for this channel by counting my own
        // outbound routing decisions during dispatch. Count "distinct valid non-local packed
        // entries per T" — same rule works for reduce and expand modes because of how
        // dispatch fills the map.
        //
        // Concrete example — token T=0 on channel C, topK=4, my scaleout_rank_idx=3,
        // routing (k=0 → scaleout 1, k=1 → scaleout 0, k=2 → scaleout 0, k=3 → scaleout 3):
        //
        //   Reduce mode (map = [ pack(1,0,C), pack(0,0,C), pack(0,0,C), pack(3,0,C) ]):
        //     k=0: valid, rank=1 (remote), first sight of pack(1,0,C)         → count
        //     k=1: valid, rank=0 (remote), first sight of pack(0,0,C)         → count
        //     k=2: valid, rank=0 (remote), pack(0,0,C) already seen (dup)     → skip
        //     k=3: valid, rank=3 (== self, local-bypass, no put fires)        → skip
        //   Expected puts for T=0 = 2. Peer R=0 sends one reduced partial covering k=1+k=2.
        //
        //   Expand mode (map = [ pack(1,0,C), pack(0,0,C), pack(0,1,C), pack(3,0,C) ]):
        //     k=0: valid, rank=1 (remote), unique packed value  → count
        //     k=1: valid, rank=0 (remote), unique packed value  → count
        //     k=2: valid, rank=0 (remote), unique packed value  → count
        //     k=3: valid, rank=3 (self)                          → skip
        //   Expected puts for T=0 = 3. Peer R=0 sends two separate partials (k=1, k=2)
        //   at distinct slots because each valid k has its own packed slot in expand mode.
        int local_expected_count_per_rank[kNumScaleoutRanks] = {};
        {
            #pragma unroll
            for (int t_in_channel = lane_idx; t_in_channel < kNumMaxTokensPerChannel; t_in_channel += 32) {
                const int token_idx = channel_idx + t_in_channel * kNumChannels;
                if (token_idx >= num_combined_tokens)
                    continue;
                // Load this token's map entries once, then dedup+count within the topK block.
                int packed_entries[kNumTopk];
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k)
                    packed_entries[k] = __ldg(token_map_at_dispatch + token_idx * kNumTopk + k);
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k) {
                    const int p = packed_entries[k];
                    if (p < 0)
                        continue;
                    int rank, slot, channel;
                    unpack_combine_recv_addr(p, rank, slot, channel);
                    // Sanity: token T dispatched by us with `T % kNumChannels == channel_idx`
                    // must have its map entry's channel field equal to channel_idx.
                    EP_DEVICE_ASSERT(channel == channel_idx);
                    if (rank == scaleout_rank_idx)
                        continue;   // local-bypass, no RDMA put
                    bool is_duplicate = false;
                    #pragma unroll
                    for (int prior_k = 0; prior_k < k; ++ prior_k)
                        is_duplicate |= (packed_entries[prior_k] == p);
                    if (not is_duplicate)
                        local_expected_count_per_rank[rank] += 1;
                }
            }
        }

        // Each lane in the Forward warp (= channel) accumulated the expected tokens per scale-out
        // rank in its own slice of the map. All-reduce across lanes so every lane sees
        // the full per-rank total. Then divide by the batch size, that's the aggregation
        // granularity we expect on the wire.
        int num_expected_arrivals = 0;
        #pragma unroll
        for (int r = 0; r < kNumScaleoutRanks; ++ r) {
            const int rank_total = ptx::reduce_add(local_expected_count_per_rank[r]);
            if constexpr (kSmallBatch and kPiecesPerToken > 0) {
                // Same pieces as the proxy posts: full batches, then a shorter last batch
                const int token_bytes = token_layout.get_num_bytes<false>();
                const int piece_bytes = sb_piece_bytes(token_bytes);
                const int num_full = rank_total / kBatchSize, num_last = rank_total % kBatchSize;
                num_expected_arrivals += num_full * math::ceil_div(kBatchSize * token_bytes, piece_bytes) +
                                         (num_last > 0 ? math::ceil_div(num_last * token_bytes, piece_bytes) : 0);
            } else {
                num_expected_arrivals += math::ceil_div(rank_total, kBatchSize) * (kSmallBatch ? EP_SB_PUT_SPLIT : 1);
            }
        }
        __syncwarp();
        // Wait for the per-channel indexed signal to accumulate `num_expected_arrivals`
        // increments from the remote senders. Bump the shadow by the expected delta to
        // get the target count, then poll the actual signal until it catches up. only
        // one lane polls. We poll with a timeout (instead of the blocking
        // `waitSignalMeetShadow`) so a stuck peer surfaces a diagnostic rather than
        // hanging.
        if (ptx::elect_one_sync()) {
            const auto shadow_ptr = gin.gin.getSignalShadowPtr(signal_id);
            const auto target = (*shadow_ptr += static_cast<uint64_t>(num_expected_arrivals));
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = gin.gin.readSignal(signal_id, comm::kNumSignalBits, cuda::memory_order_acquire);
                if (comm::signal_delta(signal, target) >= 0)
                    return true;

                if (is_last_check) {
                    printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                           "channel: %d, signal: %lu, target: %lu\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           channel_idx, signal, target);
                }
                return false;
            });
#ifdef EP_SB_PROBE
            atomicMax(&sbc_probe[sm_idx][5], ptx::globaltimer());
#endif
        }
        __syncwarp();
#ifdef EP_SB_PROBE
        if (lane_idx == 0) {
            atomicMax(&sbc_probe[sm_idx][14], ptx::globaltimer());   // slot 14: forward warp exit
            sbc_chan[channel_idx][5] = ptx::globaltimer();
        }
#endif
        }
    } else if (kTailPublisher and warp_idx == kTailPublisherWarpIdx) {
        // Lane l publishes scale-up warp l's tails: acquire its sequence (CTA scope), read its
        // counts, then one system-scope fence for all lanes and relaxed stores to the peers
        // (fence + strong store is a release pattern, cumulative over what the lane acquired)
        const bool owns = lane_idx < kNumScaleupWarps;
        const int w = owns ? lane_idx : 0;
        const int pub_channel_idx = sm_idx * kNumChannelsPerSM + w % kNumChannelsPerSM;
        int last_seq = 0;
        bool finished = not owns;
        int last_pub[kNumScaleupRanks];
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanks; ++ j)
            last_pub[j] = 0;
        while (true) {
            int seq = last_seq;
            bool done = false;
            if (owns and not finished) {
                done = ptx::ld_acquire_cta(pub_done + w) != 0;
                seq = ptx::ld_acquire_cta(pub_seq + w);
            }
            const bool changed = seq != last_seq;
            if (__any_sync(0xffffffffu, changed)) {
                int counts[kNumScaleupRanks];
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanks; ++ j)
                    counts[j] = changed ? pub_counts[w * kNumScaleupRanks + j] : last_pub[j];
#ifdef EP_SB_PROBE
                const auto probe_fence_t0 = ptx::globaltimer();
#endif
                ptx::fence_acq_rel_sys();
#ifdef EP_SB_PROBE
                if (lane_idx == 0) {
                    const auto t = ptx::globaltimer();
                    atomicAdd(&sbc_probe[sm_idx][16], 1ull);
                    atomicAdd(&sbc_probe[sm_idx][17], t - probe_fence_t0);
                    atomicMax(&sbc_probe[sm_idx][18], t);
                }
#endif
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanks; ++ j) {
                    if (changed and counts[j] != last_pub[j]) {
                        ptx::st_relaxed_sys(gin.get_sym_ptr<ncclTeamTagLsa>(scaleup_tail_ptr(pub_channel_idx, scaleup_rank_idx), j), counts[j]);
                        last_pub[j] = counts[j];
                    }
                }
                last_seq = seq;
            } else {
                __nanosleep(64);
            }
            // `done` was read before `seq`, so the sequence read covers the final counts
            if (owns and done and seq == last_seq)
                finished = true;
            if (__all_sync(0xffffffffu, finished))
                break;
        }
        __syncwarp();
    } else if constexpr (kSmallBatch and EP_SB_PROXY_MODE == 1) {
        // Small-batch proxy, one warp per SM: lane `f` tracks ring `f`; every sweep collects the
        // batches waiting in all rings and posts them with one lane each, so the lanes of a QP
        // form one coalesced GIN post (one doorbell) instead of one post per batch. Batches that
        // arrive during a sweep go into the next one.
        if (warp_idx == kNumDataWarps) {
            const bool owns_ring = lane_idx < kNumForwardWarps;
            const int f = owns_ring ? lane_idx : 0;
            unsigned tail = 0;
            bool ring_done = not owns_ring;
            while (true) {
                unsigned avail = 0;
                bool fwd_done = ring_done;
                if (not ring_done) {
                    // `done` is published after the last head update, so read it first
                    const bool done = ptx::ld_acquire_cta(proxy_ring_layout.get_done(f)) != 0;
                    const unsigned head = ptx::ld_acquire_cta(proxy_ring_layout.get_head(f));
                    avail = head - tail;
                    ring_done = done and avail == 0;
                    fwd_done = done;
                }
                if (__all_sync(0xffffffffu, ring_done))
                    break;
                const unsigned total = static_cast<unsigned>(ptx::reduce_add(static_cast<int>(avail)));
                if (total == 0)
                    continue;
                if constexpr (EP_SB_PROXY_MIN > 1) {
                    if (total < static_cast<unsigned>(EP_SB_PROXY_MIN) and not __all_sync(0xffffffffu, fwd_done))
                        continue;
                }
                unsigned excl = avail;
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    const unsigned v = __shfl_up_sync(0xffffffffu, excl, d);
                    excl += lane_idx >= d ? v : 0u;
                }
                excl -= avail;
#ifdef EP_SB_PROBE
                const auto sweep_t0 = ptx::globaltimer();
                if (lane_idx == 0)
                    atomicMax(&sbc_probe[sm_idx][20], sweep_t0);
#endif
                constexpr int kPutSplit = EP_SB_PUT_SPLIT;
                EP_STATIC_ASSERT(kPutSplit >= 1, "Invalid put split");
                // With pieces sized per token, a batch of b tokens is ceil(b * token / piece) puts
                // (`kPieceBytes` below), so a short last batch posts fewer and fuller puts
                constexpr int kSlotsPerBatch = kPiecesPerToken > 0 ? kBatchSize * kPiecesPerToken : kPutSplit;
                const int token_piece_bytes = kPiecesPerToken > 0 ? sb_piece_bytes(token_layout.get_num_bytes<false>()) : 0;
                for (unsigned base = 0; base < total * kSlotsPerBatch; base += 32) {
                    const unsigned item = base + static_cast<unsigned>(lane_idx);
                    const unsigned j = item / kSlotsPerBatch;
                    const int piece = static_cast<int>(item % kSlotsPerBatch);
                    int g = -1;
                    unsigned pos = 0;
                    #pragma unroll
                    for (int q = 0; q < kNumForwardWarps; ++ q) {
                        const unsigned e = __shfl_sync(0xffffffffu, excl, q);
                        const unsigned a = __shfl_sync(0xffffffffu, avail, q);
                        const unsigned t = __shfl_sync(0xffffffffu, tail, q);
                        if (j < total and j >= e and j < e + a)
                            g = q, pos = t + (j - e);
                    }
                    if (g >= 0) {
                        const auto& desc = proxy_ring_layout.get_ring(g)[pos % static_cast<unsigned>(kProxyRingDepth)];
                        int offset = 0, num_bytes = desc.num_bytes;
                        if constexpr (kPiecesPerToken > 0) {
                            offset = piece * token_piece_bytes;
                            num_bytes = min(token_piece_bytes, desc.num_bytes - offset);
                        } else if constexpr (kPutSplit > 1) {
                            const int piece_bytes = (math::ceil_div(desc.num_bytes, kPutSplit) + 15) / 16 * 16;
                            offset = piece * piece_bytes;
                            num_bytes = min(piece_bytes, desc.num_bytes - offset);
                        }
                        // Token-sized pieces: slots past a short batch's end post nothing
                        if (num_bytes > 0) {
                            const auto [put_qp_idx, put_sharing_mode] =
                                comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, g);
                            const handle::NCCLGin put_gin(nccl_dev_comm, nccl_window, put_qp_idx, put_sharing_mode);
                            put_gin.put<ncclTeamTagRail>(
                                math::advance_ptr(desc.recv_ptr, offset), math::advance_ptr(desc.send_ptr, offset),
                                num_bytes, desc.dst, 0,
                                ncclGin_SignalAdd{channel_signal_id(g), static_cast<uint64_t>(1)});
                        }
                    }
                    __syncwarp();
                }
                if (owns_ring and avail > 0) {
                    tail += avail;
                    ptx::st_release_cta(proxy_ring_layout.get_tail(f), tail);
                }
                __syncwarp();
#ifdef EP_SB_PROBE
                if (lane_idx == 0) {
                    const auto t = ptx::globaltimer();
                    atomicMax(&sbc_probe[sm_idx][4], t);
                    atomicMax(&sbc_probe[sm_idx][8], ~0ull - sweep_t0);
                    atomicAdd(&sbc_probe[sm_idx][9], 1ull);
                    atomicAdd(&sbc_probe[sm_idx][10], t - sweep_t0);
                    atomicAdd(&sbc_probe[sm_idx][11], static_cast<unsigned long long>(total));
                }
#endif
            }
            if constexpr (EP_SB_COMBINE_FLUSH != 0) {
                if (lane_idx < kNumForwardWarps) {
                    const auto [flush_qp_idx, flush_sharing_mode] =
                        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, lane_idx);
                    const handle::NCCLGin flush_gin(nccl_dev_comm, nccl_window, flush_qp_idx, flush_sharing_mode);
                    comm::flush_with_timeout<kNumTimeoutCycles>(flush_gin.gin, ncclCoopThread());
                }
            }
        }
        __syncwarp();
#ifdef EP_SB_PROBE
        if (lane_idx == 0)
            atomicMax(&sbc_probe[sm_idx][6], ptx::globaltimer());
#endif
    } else if constexpr (kSmallBatch) {
        // Small-batch proxy, one warp per channel: drains its forward warp's ring and posts each
        // batch as soon as it is handed over, so the scale-out puts overlap the reduce phase
        const int ring_idx = warp_idx - kNumDataWarps;
        if (ptx::elect_one_sync()) {
            const auto signal_id = channel_signal_id(ring_idx);
            ProxyPutDesc* const ring = proxy_ring_layout.get_ring(ring_idx);
            unsigned tail = 0;
            while (true) {
                const bool done = ptx::ld_acquire_cta(proxy_ring_layout.get_done(ring_idx)) != 0;
                const unsigned head = ptx::ld_acquire_cta(proxy_ring_layout.get_head(ring_idx));
                if (head == tail) {
                    if (done)
                        break;
                    continue;
                }
#ifdef EP_SB_PROBE
                const auto sweep_t0 = ptx::globaltimer();
                const unsigned max_pending = head - tail;
#endif
                for (; tail != head; ++ tail) {
                    const auto& desc = ring[tail % static_cast<unsigned>(kProxyRingDepth)];
                    gin.put<ncclTeamTagRail>(
                        desc.recv_ptr, desc.send_ptr,
                        desc.num_bytes,
                        desc.dst,
                        0, /*flags=0*/
                        ncclGin_SignalAdd{signal_id, static_cast<uint64_t>(1)}
                    );
                }
                ptx::st_release_cta(proxy_ring_layout.get_tail(ring_idx), tail);
#ifdef EP_SB_PROBE
                {
                    const auto t = ptx::globaltimer();
                    atomicMax(&sbc_probe[sm_idx][4], t);
                    atomicMax(&sbc_probe[sm_idx][8], ~0ull - sweep_t0);
                    atomicAdd(&sbc_probe[sm_idx][9], 1ull);
                    atomicAdd(&sbc_probe[sm_idx][10], t - sweep_t0);
                    atomicAdd(&sbc_probe[sm_idx][11], max_pending);
                }
#endif
            }
            if constexpr (EP_SB_COMBINE_FLUSH != 0)
                comm::flush_with_timeout<kNumTimeoutCycles>(gin.gin, ncclCoopThread());
        }
        __syncwarp();
#ifdef EP_SB_PROBE
        if (lane_idx == 0)
            atomicMax(&sbc_probe[sm_idx][6], ptx::globaltimer());
#endif
    } else {
        // Count the arrivals this SM must wait for.
        //
        // The rule is "distinct valid non-local packed entries per token", scanned out of
        // `token_map_at_dispatch` -- my own outbound routing decisions during dispatch, which are
        // exactly what the peers send back. Summed over all `kNumChannelsPerSM` channels of this SM.
        // Full warp, before the electing thread takes over.
        //
        // Concrete example -- token T=0 on channel C, topK=4, my scaleout_rank_idx=3, routing
        // (k=0 -> scaleout 1, k=1 -> scaleout 0, k=2 -> scaleout 0, k=3 -> scaleout 3):
        //   Reduce mode (map = [pack(1,0,C), pack(0,0,C), pack(0,0,C), pack(3,0,C)]):
        //     k=0 remote, first sight          -> count
        //     k=1 remote, first sight          -> count
        //     k=2 remote, duplicate of k=1     -> skip
        //     k=3 == self, local bypass        -> skip
        //   Expected puts for T=0 = 2; peer 0 sends one reduced partial covering k=1+k=2.
        int sm_num_expected_arrivals = 0;
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");
        #pragma unroll
        for (int c = 0; c < kNumChannelsPerSM; ++ c) {
            const int scan_channel_idx = sm_idx * kNumChannelsPerSM + c;
            int local_expected_count_per_rank[kNumScaleoutRanks] = {};
            for (int t_in_channel = lane_idx; t_in_channel < kNumMaxTokensPerChannel; t_in_channel += 32) {
                const int token_idx = scan_channel_idx + t_in_channel * kNumChannels;
                if (token_idx >= num_combined_tokens)
                    continue;
                int packed_entries[kNumTopk];
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k)
                    packed_entries[k] = __ldg(token_map_at_dispatch + token_idx * kNumTopk + k);
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k) {
                    const int p = packed_entries[k];
                    if (p < 0)
                        continue;
                    int rank, slot, channel;
                    unpack_combine_recv_addr(p, rank, slot, channel);
                    EP_DEVICE_ASSERT(channel == scan_channel_idx);
                    if (rank == scaleout_rank_idx)
                        continue;   // local bypass, no RDMA put comes back for it
                    bool is_duplicate = false;
                    #pragma unroll
                    for (int prior_k = 0; prior_k < k; ++ prior_k)
                        is_duplicate |= (packed_entries[prior_k] == p);
                    if (not is_duplicate)
                        local_expected_count_per_rank[rank] += 1;
                }
            }
            // Reduce across lanes BEFORE the ceil_div: batching happens per (peer, channel), not
            // per lane, so dividing a lane's partial count would round up far too often.
            #pragma unroll
            for (int r = 0; r < kNumScaleoutRanks; ++ r)
                sm_num_expected_arrivals +=
                    math::ceil_div(ptx::reduce_add(local_expected_count_per_rank[r]), kBatchSize);
        }
        __syncwarp();

        if (ptx::elect_one_sync()) {
            int num_forward_warps_done_cnt = 0;
            unsigned tail[kNumForwardWarps] = {};
            bool ring_done[kNumForwardWarps] = {};
            // A single thread in a single warp posts for the whole SM, using one GIN context
            // and one indexed signal.
            const auto rail_signal_id = static_cast<ncclGinSignal_t>(
                comm::get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, 0));

            // Either THREAD or GPU. One lane in one warp posts, so CTA is unnecessary and
            // THREAD is more efficient.
            const auto [rail_qp, mapped_mode] =
                comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, 0);
            const auto rail_mode = mapped_mode == NCCL_GIN_RESOURCE_SHARING_GPU
                                 ? NCCL_GIN_RESOURCE_SHARING_GPU
                                 : NCCL_GIN_RESOURCE_SHARING_THREAD;
            handle::NCCLGin rail_ctx(nccl_dev_comm, nccl_window, rail_qp, rail_mode);

            while (num_forward_warps_done_cnt < kNumForwardWarps) {
                // Two passes, so the round's last put -- the one that rings the doorbell -- is
                // known before any put is issued.

                // Pass 1: count what each ring holds.
                int pending_count[kNumForwardWarps] = {};
                int total_pending = 0;
                // Once any warp is done, stop holding rounds back and drain every sweep.
                bool any_ring_finished = false;

                #pragma unroll
                for (int forward_warp_idx = 0; forward_warp_idx < kNumForwardWarps; ++ forward_warp_idx) {
                    if (ring_done[forward_warp_idx]) {
                        any_ring_finished = true;
                        continue;
                    }

                    const unsigned head = ptx::ld_acquire_cta(proxy_ring_layout.get_head(forward_warp_idx));

                    if (tail[forward_warp_idx] == head) {
                        if (ptx::ld_acquire_cta(proxy_ring_layout.get_done(forward_warp_idx)) == 1 and
                            tail[forward_warp_idx] == ptx::ld_acquire_cta(proxy_ring_layout.get_head(forward_warp_idx))) {
                            ring_done[forward_warp_idx] = true;
                            ++ num_forward_warps_done_cnt;
                            any_ring_finished = true;
                        }
                        continue;
                    }

                    // Unsigned subtraction, correct across wrap; bounded by kProxyRingDepth.
                    pending_count[forward_warp_idx] = static_cast<int>(head - tail[forward_warp_idx]);
                    total_pending += pending_count[forward_warp_idx];

                    // A non-empty ring can still have a finished producer.
                    if (ptx::ld_acquire_cta(proxy_ring_layout.get_done(forward_warp_idx)) != 0)
                        any_ring_finished = true;
                }

                // Wait for one descriptor per forward warp, then send them all under one
                // doorbell. A floor, not a cap. Exempt in the tail, which would wait forever.
                const bool drain_now = total_pending > 0 and
                                       (total_pending >= kNumForwardWarps or any_ring_finished);

                const auto issue_put = [&](const ProxyPutDesc& d, const int& ring_idx,
                                           const bool& ring_doorbell) {
                    rail_ctx.put<ncclTeamTagRail>(
                        d.recv_ptr, d.send_ptr,
                        d.num_bytes,
                        d.dst,
                        ring_doorbell ? 0 : ncclGinOptFlagsAggregateRequests,
                        ncclGin_SignalAdd{rail_signal_id, static_cast<uint64_t>(1)}
                    );
                    ++ tail[ring_idx];
                    ptx::st_release_cta(proxy_ring_layout.get_tail(ring_idx), tail[ring_idx]);
                };

                if (not drain_now) {
                    __nanosleep(1000);
                    continue;
                }

                // Pass 2: drain the whole snapshot. Anything queued mid-round waits for the next.
                //
                // Issuing lags one descriptor behind the scan so each put knows the next put's
                // destination. A deferred doorbell only covers one destination -- GDAKI gives each
                // peer its own QP, hence its own doorbell -- so a destination change must ring.
                ProxyPutDesc pend{};
                int pend_ring = -1;
                #pragma unroll
                for (int forward_warp_idx = 0; forward_warp_idx < kNumForwardWarps; ++ forward_warp_idx) {
                    // Fixed base: `issue_put` advances `tail[forward_warp_idx]` for the descriptor
                    // held back, which trails this scan by one.
                    const unsigned read_base = tail[forward_warp_idx];
                    for (int k = 0; k < pending_count[forward_warp_idx]; ++ k) {
                        const ProxyPutDesc desc = proxy_ring_layout.get_ring(forward_warp_idx)[
                            (read_base + static_cast<unsigned>(k)) % static_cast<unsigned>(kProxyRingDepth)];
                        if (pend_ring >= 0)
                            issue_put(pend, pend_ring, pend.dst != desc.dst);
                        pend = desc;
                        pend_ring = forward_warp_idx;
                    }
                }
                // `drain_now` guarantees a descriptor was scanned. The round's last put always
                // rings, so no WQE is left un-rung.
                issue_put(pend, pend_ring, true);
            }

            // Wait for all arrivals from the peer rail SMs.
            const auto shadow_ptr = rail_ctx.gin.getSignalShadowPtr(rail_signal_id);
            const auto target = (*shadow_ptr += static_cast<uint64_t>(sm_num_expected_arrivals));
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal =
                    rail_ctx.gin.readSignal(rail_signal_id, comm::kNumSignalBits, cuda::memory_order_acquire);
                if (comm::signal_delta(signal, target) >= 0)
                    return true;
                if (is_last_check) {
                    // SM-wide, not per channel: one signal now covers every channel of this SM, so
                    // the per-channel attribution the old message printed no longer exists.
                    printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, "
                           "scale-up: %d/%d, sm: %d, signal: %lu, target: %lu, expected: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           sm_idx, signal, target, sm_num_expected_arrivals);
                }
                return false;
            });

            comm::flush_with_timeout<kNumTimeoutCycles>(rail_ctx.gin, ncclCoopThread());
        }
        __syncwarp();
    }

    // No barrier at epilogue
#ifdef EP_SB_PROBE
    if constexpr (kSmallBatch) {
        __syncthreads();
        if (sm_idx == 0 and thread_idx == 0 and (combine_iteration & (EP_SB_PROBE_EVERY - 1)) <= 1) {
            // Give the other blocks a moment to land their stamps (no grid sync here by design)
            const auto t_wait = ptx::globaltimer();
            while (ptx::globaltimer() - t_wait < 20000ull) {}
            unsigned long long q[28] = {~0ull};
            for (int b = 0; b < kNumSMs; ++ b) {
                q[0] = min(q[0], sbc_probe[b][0]);
                #pragma unroll
                for (int i = 1; i < 6; ++ i)
                    q[i] = max(q[i], sbc_probe[b][i]);
                q[6] = max(q[6], sbc_probe[b][6]);
                q[7] = max(q[7], sbc_probe[b][7]);
                #pragma unroll
                for (int i = 8; i < 28; ++ i)
                    q[i] = (i == 8 or i == 14 or i == 18 or i == 19 or i == 20 or i == 21 or i == 25) ? max(q[i], sbc_probe[b][i]) : q[i] + sbc_probe[b][i];
            }
            printf("SBC ex=%d so=%d su=%d it=%d entry=%llu sudone=%llu firsttail=%llu reduced=%llu issued=%llu inbound=%llu "
                   "proxyexit=%llu max_su_tokens=%llu firstsweep=%llu sweeps=%llu sweep_ns=%llu puts=%llu "
                   "waitns=%llu reducens=%llu exit=%llu ntok=%llu pubrounds=%llu pubfencens=%llu lastpub=%llu pass0end=%llu lastsweep=%llu pass0first=%llu p0tok=%llu p0waitns=%llu p0reducens=%llu supass0=%llu\n",
                   static_cast<int>(kUseExpandedLayout), scaleout_rank_idx, scaleup_rank_idx, combine_iteration,
                   q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], ~0ull - q[8], q[9], q[10], q[11],
                   q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], ~0ull - q[21], q[22], q[23], q[24], q[25]);
            if (scaleup_rank_idx == 0) {
                for (int c = 0; c < kNumChannels; ++ c) {
                    const auto* v = sbc_chan[c];
                    printf("SBCH ex=%d so=%d it=%d ch=%d p0tok=%llu p0first=%lld p0end=%lld p0wait=%llu p0red=%llu p1tok=%llu fexit=%lld\n",
                           static_cast<int>(kUseExpandedLayout), scaleout_rank_idx, combine_iteration, c,
                           v[0], v[0] ? static_cast<long long>(v[6] - q[0]) : -1ll, static_cast<long long>(v[1] - q[0]),
                           v[2], v[3], v[4], static_cast<long long>(v[5] - q[0]));
                }
                for (int b = 0; b < kNumSMs; ++ b) {
                    for (int w = 0; w < kNumScaleupWarps; ++ w) {
                        const auto* v = sbc_su[b * 32 + w];
                        printf("SBSU ex=%d so=%d it=%d sm=%d w=%d rows0=%llu end0=%lld rows=%llu end=%lld\n",
                               static_cast<int>(kUseExpandedLayout), scaleout_rank_idx, combine_iteration, b, w,
                               v[0], static_cast<long long>(v[1] - q[0]), v[2], static_cast<long long>(v[3] - q[0]));
                    }
                }
            }
#ifdef EP_GDA_PROBE
            printf("GDAC ex=%d so=%d su=%d it=%d posts=%llu bp=%llu wqe=%llu rv=%llu db=%llu post=%llu setup=%llu\n",
                   static_cast<int>(kUseExpandedLayout), scaleout_rank_idx, scaleup_rank_idx, combine_iteration,
                   nccl::gin::efa_gda::ep_gda_probe[0], nccl::gin::efa_gda::ep_gda_probe[1], nccl::gin::efa_gda::ep_gda_probe[2],
                   nccl::gin::efa_gda::ep_gda_probe[3], nccl::gin::efa_gda::ep_gda_probe[4], nccl::gin::efa_gda::ep_gda_probe[5],
                   nccl::gin::efa_gda::ep_gda_probe[6]);
#endif
        }
    }
#endif
}

}  // namespace deep_ep::elastic
