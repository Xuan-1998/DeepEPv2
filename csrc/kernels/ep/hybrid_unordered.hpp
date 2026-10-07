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

#include <algorithm>
#include <cstdio>
#include <format>
#include <memory>
#include <string>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/impls/ep/unordered/gin.cuh>
#include <deep_ep/impls/ep/unordered/layout.cuh>
#include <deep_ep/impls/ep/unordered/proxy_ring.cuh>

#include "../../runtime/jit.hpp"
#include "../../utils/system.hpp"
#include "hybrid_kernel.hpp"

namespace deep_ep::ep {

// Hybrid kernels for GIN backends without ordered delivery or strong/VA signals.
//
// The default hybrid kernels publish a tail through a trailing signal and assume that all
// data written before the signal has landed. This pair instead sends each channel's tokens
// as a few batched puts, each carrying an in-band header and completing a counting signal:
// the receiver treats the signal as a completion count and validates every batch through
// its header, so correctness does not depend on the order in which puts land. GIN contexts
// are shared across SMs and only counting (indexed) signals are requested, which is what
// the EFA GDA backend provides. The combine returns partials per channel through batched
// puts with a shared counting signal, and the reduce epilogue locates them through a
// per-(token, k) receive map that dispatch records (the variant's dispatch handle).
class UnorderedHybridKernel final: public HybridKernelVariant {
    // In-band header iteration, so a receiver can tell this dispatch's batches from stale ones
    mutable int dispatch_iteration = 0;

    // Per-context indexed-signal budget resolved at construction
    int gin_indexed_signals_cnt = 0;

public:
    const char* name() const override {
        return "unordered";
    }

    comm::GinRequirements get_gin_requirements(const int& num_allocated_qps, const int& num_rdma_ranks) override {
        // One GIN context supplies one QP; fewer contexts leave more indexed signals per context
        const int requested = num_allocated_qps == 0 ? unordered::kDefaultGinContextCnt : num_allocated_qps;
        const int num_contexts = std::clamp(requested, unordered::kMinGinContextCnt, unordered::kMaxGinContextCnt);
        if (num_contexts != requested)
            printf("[WARN] DeepEP clamped num_allocated_qps from %d to %d: the unordered GIN layout "
                   "supports [%d, %d] contexts (one GIN context supplies one QP)\n",
                   requested, num_contexts, unordered::kMinGinContextCnt, unordered::kMaxGinContextCnt);

        // Single-node communicators never touch the scale-out path and need no indexed signals
        gin_indexed_signals_cnt = num_rdma_ranks > 1 ? unordered::gin_indexed_signals_for(num_contexts) : 0;
        EP_HOST_ASSERT((num_rdma_ranks <= 1 or gin_indexed_signals_cnt >= num_rdma_ranks) and
                       "GIN indexed-signal budget cannot give the barrier one signal per scale-out rank; "
                       "reduce num_allocated_qps to raise the per-context signal count");
        if (get_env<int>("EP_BUFFER_DEBUG"))
            printf("GIN layout: gin_context_cnt=%d, gin_indexed_signals_cnt=%d\n", num_contexts, gin_indexed_signals_cnt);
        return {
            .context_count = num_contexts,
            .signal_count = gin_indexed_signals_cnt,
            .exclusive_contexts = false,
            .strong_signals_required = false,
            .va_signals_required = false,
        };
    }

    int64_t get_dispatch_scaleout_buffer_size(const layout::TokenLayout& token_layout,
                                              const int& num_max_tokens_per_rank,
                                              const int& num_scaleout_ranks,
                                              const int& num_max_channels) const override {
        // Send and receive buffers are both per peer and per channel part
        const auto slot_layout = unordered::ScaleoutSlotLayout(token_layout);
        const int num_slots = num_max_tokens_per_rank + num_max_channels * unordered::kScaleoutSlotRoundingReserve;
        return 2 * unordered::ScaleoutBufferLayout(slot_layout, num_scaleout_ranks, num_slots).get_num_bytes();
    }

    int64_t get_combine_scaleout_buffer_size(const layout::TokenLayout& token_layout,
                                             const int& num_max_tokens_per_rank,
                                             const int& num_topk,
                                             const int& num_scaleout_ranks,
                                             const int& num_max_channels,
                                             const bool& allow_multiple_reduction) const override {
        // Partials are packed per peer and per channel, one slot per returned partial
        const int num_slots = (num_max_tokens_per_rank + num_max_channels) * (allow_multiple_reduction ? 1 : num_topk);
        return 2 * layout::BufferLayout<false>(token_layout, num_scaleout_ranks, num_slots).get_num_bytes();
    }

    int get_num_channels_per_sm(const int& num_channels_per_sm,
                                const int& num_sms, const int& num_qps,
                                const int& num_smem_bytes, const int& num_notify_smem_bytes,
                                const layout::TokenLayout& dispatch_token_layout,
                                const layout::TokenLayout& combine_token_layout,
                                const bool& prefer_overlap_with_compute) const override {
        // The dispatch kernel double-buffers the forward warp's TMA loads unless communication
        // overlaps with compute, so a channel may need three TMA buffers instead of two
        const int dispatch_buffers_per_channel = prefer_overlap_with_compute
            ? unordered::kNumDispatchSendBuffers + 1
            : unordered::kNumDispatchBuffersPerChannel;
        const int num_token_buffers = std::min({
            (num_smem_bytes - num_notify_smem_bytes) / dispatch_token_layout.get_num_bytes<true>(),
            32 - kNumNotifyWarps,
            num_smem_bytes / combine_token_layout.get_num_bytes<true>()});
        int result = std::min(num_token_buffers / dispatch_buffers_per_channel, num_channels_per_sm);

        // Fit the indexed-signal budget: every channel part needs its own signal on its context.
        // `with_notify` is pinned to true so a cached dispatch derives the same count the
        // handle was shaped with.
        if (gin_indexed_signals_cnt > 0)
            result = unordered::constexpr_channels_per_sm(gin_indexed_signals_cnt, num_sms, num_qps, true, result);

        // The combine carves the proxy hand-off rings out of the same dynamic shared memory as
        // its TMA buffers (see `launch_combine`)
        while (result > 1 and
               static_cast<int64_t>(2 * result) * combine_token_layout.get_num_bytes<true, int64_t>() +
                   unordered::ProxyRingLayout::get_num_bytes(result, unordered::kProxyRingDepthDefault) > num_smem_bytes)
            -- result;
        EP_HOST_ASSERT(result >= 1 and "shared memory cannot host a single channel at this token size");
        if (get_env<int>("EP_BUFFER_DEBUG") and result != num_channels_per_sm)
            printf("Unordered hybrid kernels reduce channels per SM from %d to %d\n", num_channels_per_sm, result);
        return result;
    }

    // The receive map: `[num_max_tokens_per_rank, num_topk]` packed `(rank, slot, channel)` entries
    bool has_dispatch_handle() const override {
        return true;
    }

    torch::Tensor make_dispatch_handle(const int& num_max_tokens_per_rank, const int& num_topk) const override {
        return torch::empty({num_max_tokens_per_rank, num_topk},
                            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt));
    }

    void check_dispatch_handle(const torch::Tensor& handle, const int& num_max_tokens_per_rank, const int& num_topk) const override {
        EP_HOST_ASSERT(handle.dim() == 2 and handle.size(0) == num_max_tokens_per_rank and handle.size(1) == num_topk);
        EP_HOST_ASSERT(handle.is_cuda() and handle.is_contiguous());
        EP_HOST_ASSERT(handle.scalar_type() == torch::kInt);
    }

    void launch_dispatch(const HybridDispatchArgs& args, const HybridDispatchExtras& extras,
                         const at::cuda::CUDAStream& stream) const override {
        EP_HOST_ASSERT(extras.handle != nullptr);
        const int num_notify_warps = args.cached_mode ? 0 : kNumNotifyWarps;
        const int num_scaleout_warps = args.num_channels_per_sm, num_forward_warps = args.num_channels_per_sm;
        const int num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;

        // Compile
        const auto kernel = jit->compile("dispatch", std::format(R"(
#include <deep_ep/impls/ep/unordered/dispatch.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::unordered::dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
            args.do_cpu_sync,
            /* reuse_slot_indices */ args.cached_mode,
            extras.allow_multiple_reduction,
            extras.do_expand,
            /* double_buffer_forward */ not extras.prefer_overlap_with_compute,
            args.num_sms,
            num_notify_warps, num_scaleout_warps, num_forward_warps,
            args.num_scaleout_ranks, args.num_scaleup_ranks,
            args.hidden * args.elem_size, args.num_sf_packs,
            args.num_max_tokens_per_rank,
            args.num_experts, args.num_topk, args.expert_alignment,
            args.num_qps, args.num_timeout_cycles,
            gin_indexed_signals_cnt));

        // Launch
        ++ dispatch_iteration;
        jit->launch(
            kernel, {
                .stream = stream.stream(),
                .num_smem_bytes = args.num_smem_bytes,
                .grid_dim = dim3(args.num_sms, 1, 1),
                .block_dim = dim3(num_threads, 1, 1),
                .cluster_dim = dim3(2 - (args.num_sms % 2), 1, 1),
                .cooperative = true,
            },
            args.x, static_cast<sf_pack_t*>(args.sf), args.topk_idx, args.topk_weights,
            args.cumulative_local_expert_recv_stats,
            args.psum_num_recv_tokens_per_scaleup_rank,
            args.psum_num_recv_tokens_per_expert,
            args.num_unaligned_recv_tokens_per_expert,
            args.dst_buffer_slot_idx,
            args.token_metadata_at_forward,
            extras.handle,
            args.num_tokens,
            args.sf_token_stride, args.sf_hidden_stride,
            args.nccl_dev_comm, args.nccl_window,
            args.buffer,
            args.workspace, args.mapped_host_workspace,
            args.scaleout_rank_idx, args.scaleup_rank_idx,
            dispatch_iteration
        );
    }

    void* launch_combine(const HybridCombineArgs& args, const HybridCombineExtras& extras,
                         const at::cuda::CUDAStream& stream) const override {
        EP_HOST_ASSERT(extras.handle != nullptr);
        EP_HOST_ASSERT(args.num_channels % args.num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        const auto token_layout = get_combine_token_layout(args.hidden, sizeof(nv_bfloat16), args.num_topk);

        // One scale-up and one forward warp per channel, plus a proxy warp that issues the puts
        const int num_scaleup_warps = args.num_channels / args.num_sms, num_forward_warps = num_scaleup_warps;
        const int num_data_warps = num_scaleup_warps + num_forward_warps;
        const int num_threads = (num_data_warps + 1) * 32;
        EP_HOST_ASSERT(num_threads <= 1024 and
                       "combine warp count (scale-up + forward + proxy) exceeds the 1024-thread block limit; "
                       "use at least num_channels / 15 SMs");

        // TMA buffers and the proxy hand-off rings share the dynamic shared memory; the channel
        // budget in `get_num_channels_per_sm` keeps both within it
        EP_HOST_ASSERT(static_cast<int64_t>(num_data_warps) * token_layout.get_num_bytes<true>() +
                       unordered::ProxyRingLayout::get_num_bytes(num_forward_warps, unordered::kProxyRingDepthDefault) <=
                       args.num_smem_bytes and
                       "Combine TMA buffers + proxy rings exceed per-block shared memory");

        // Compile
        const auto kernel = jit->compile("combine", std::format(R"(
#include <deep_ep/impls/ep/unordered/combine.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::unordered::combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
            args.use_expanded_layout, args.allow_multiple_reduction,
            args.num_sms,
            num_scaleup_warps, num_forward_warps,
            args.num_scaleout_ranks, args.num_scaleup_ranks,
            args.hidden,
            args.num_max_tokens_per_rank,
            args.num_experts,
            args.num_topk,
            args.num_qps,
            args.num_timeout_cycles));

        // Launch
        jit->launch(
            kernel, {
                .stream = stream.stream(),
                .num_smem_bytes = args.num_smem_bytes,
                .grid_dim = dim3(args.num_sms, 1, 1),
                .block_dim = dim3(num_threads, 1, 1),
                .cluster_dim = dim3(2 - (args.num_sms % 2), 1, 1),
                .cooperative = true,
            },
            static_cast<nv_bfloat16*>(args.x), static_cast<float*>(args.topk_weights),
            args.src_metadata,
            args.psum_num_recv_tokens_per_scaleup_rank,
            args.token_metadata_at_forward,
            args.channel_linked_list,
            extras.handle,
            args.nccl_dev_comm, args.nccl_window,
            args.buffer, args.workspace,
            args.scaleout_rank_idx, args.scaleup_rank_idx,
            args.num_reduced_tokens,
            extras.num_combined_tokens
        );

        // The epilogue reduces the scale-out receive buffer, which follows the scale-up buffer
        const bool is_scaleup_buffer_rank_layout =
            args.allow_multiple_reduction ? (args.num_scaleup_ranks <= args.num_topk) : false;
        const auto scaleup_buffer = layout::BufferLayout<false>(
            token_layout,
            is_scaleup_buffer_rank_layout ? args.num_scaleup_ranks : args.num_topk,
            args.num_scaleout_ranks * args.num_max_tokens_per_rank,
            args.buffer);
        return scaleup_buffer.get_buffer_end_ptr();
    }

    void launch_combine_reduce_epilogue(const HybridCombineReduceEpilogueArgs& args,
                                        const HybridCombineReduceEpilogueExtras& extras,
                                        const at::cuda::CUDAStream& stream) const override {
        EP_HOST_ASSERT(extras.handle != nullptr);

        // Maximize shared memory utilization
        // Too many warps may cause performance degrade, so we limit into 1024
        const auto token_layout = layout::TokenLayout(args.hidden * sizeof(nv_bfloat16), 0, 0, false);
        const auto num_warps = std::min<int>(args.num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
        const auto num_threads = num_warps * 32;

        // Compile
        const auto kernel = jit->compile("combine_reduce_epilogue", std::format(R"(
#include <deep_ep/impls/ep/unordered/combine_reduce_epilogue.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::unordered::combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", args.use_expanded_layout, args.allow_multiple_reduction,
            args.num_sms, num_warps,
            args.num_scaleout_ranks, args.num_scaleup_ranks,
            args.hidden,
            args.num_max_tokens_per_rank,
            args.num_experts, args.num_topk,
            extras.num_channels));

        // Launch
        jit->launch(
            kernel, {
                .stream = stream.stream(),
                .num_smem_bytes = args.num_smem_bytes,
                .grid_dim = dim3(args.num_sms, 1, 1),
                .block_dim = dim3(num_threads, 1, 1),
                .enable_pdl = true,
            },
            static_cast<nv_bfloat16*>(args.combined_x),
            args.combined_topk_weights,
            args.combined_topk_idx,
            args.reduce_buffer,
            args.bias_0, args.bias_1,
            extras.handle,
            args.num_combined_tokens,
            args.scaleout_rank_idx, args.scaleup_rank_idx
        );
    }
};

// `EP_HYBRID_KERNEL` selects the hybrid kernel pair once per process:
//   auto (default)  the default kernels on backends with ordered delivery and strong/VA
//                   signals (GDAKI and the CPU proxy), the unordered pair otherwise
//   ordered         always the default kernels
//   unordered       always the unordered pair
static std::shared_ptr<HybridKernelVariant> select_hybrid_kernel_variant(const int64_t& nccl_comm,
                                                                         const bool& allow_hybrid_mode) {
    if (not allow_hybrid_mode)
        return nullptr;

    static const std::string selection = [] {
        const auto value = get_env<std::string>("EP_HYBRID_KERNEL", "auto");
        EP_HOST_ASSERT((value == "auto" or value == "ordered" or value == "unordered") and
                       "EP_HYBRID_KERNEL must be `auto`, `ordered` or `unordered`");
        return value;
    }();

    bool use_unordered = selection == "unordered";
    if (selection == "auto") {
        const auto gin_type = comm::get_gin_type(nccl_comm, allow_hybrid_mode);
        use_unordered = gin_type != NCCL_GIN_TYPE_NONE and gin_type != NCCL_GIN_TYPE_GDAKI and gin_type != NCCL_GIN_TYPE_PROXY;
    }
    static bool printed = false;
    if (get_env<int>("EP_BUFFER_DEBUG") and not printed) {
        printf("DeepEP hybrid kernels: %s (EP_HYBRID_KERNEL=%s)\n", use_unordered ? "unordered" : "ordered", selection.c_str());
        printed = true;
    }
    return use_unordered ? std::make_shared<UnorderedHybridKernel>() : nullptr;
}

} // namespace deep_ep::ep
