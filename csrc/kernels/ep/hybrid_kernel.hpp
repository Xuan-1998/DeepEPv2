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

#include <cstdint>
#include <memory>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/layout/ep/token.cuh>
#include <deep_jit/utils/no_ref_ptr.hpp>

#include "../comm/api.hpp"

namespace deep_ep::ep {

// A hybrid (scale-out) dispatch/combine kernel pair that replaces the default one.
//
// The default hybrid kernels synchronize through a trailing signal and require a GIN
// backend with ordered delivery and strong/VA signals. A variant may implement the
// scale-out path differently, which changes everything the scale-out protocol touches:
// the GIN requirements of the device communicator, the scale-out buffer layout, the
// per-launch channel budget, the kernels themselves, and an optional per-dispatch handle
// tensor that combine reads back. The scale-up (NVLink) side, the direct kernels and the
// dispatch copy epilogue are shared and not part of this interface.
//
// `select_hybrid_kernel_variant` returns nullptr when the default kernels are in use.

// The arguments of `launch_dispatch`, `launch_combine` and `launch_combine_reduce_epilogue`
// for the hybrid path, forwarded to the variant unchanged
struct HybridDispatchArgs {
    void* x; void* sf; topk_idx_t* topk_idx; float* topk_weights;
    int* cumulative_local_expert_recv_stats;
    int* psum_num_recv_tokens_per_scaleup_rank;
    int* psum_num_recv_tokens_per_expert;
    int* num_unaligned_recv_tokens_per_expert;
    int* dst_buffer_slot_idx;
    int* token_metadata_at_forward;
    int num_tokens, num_max_tokens_per_rank;
    int hidden, elem_size;
    int num_sf_packs, sf_token_stride, sf_hidden_stride;
    int num_experts, num_topk, expert_alignment;
    deep_jit::NoRefPtr nccl_dev_comm; ncclWindow_t nccl_window;
    void* buffer; void* workspace; void* mapped_host_workspace;
    int scaleout_rank_idx, scaleup_rank_idx;
    int num_scaleout_ranks, num_scaleup_ranks;
    int num_sms, num_channels_per_sm, num_smem_bytes;
    int num_qps; int64_t num_timeout_cycles;
    bool cached_mode, do_cpu_sync;
};

struct HybridCombineArgs {
    void* x; void* topk_weights;
    int* src_metadata;
    int* psum_num_recv_tokens_per_scaleup_rank;
    int* token_metadata_at_forward;
    int* channel_linked_list;
    deep_jit::NoRefPtr nccl_dev_comm; ncclWindow_t nccl_window;
    void* buffer; void* workspace;
    int num_reduced_tokens, num_max_tokens_per_rank;
    int hidden, num_experts, num_topk;
    int num_qps; int64_t num_timeout_cycles;
    int num_scaleout_ranks, num_scaleup_ranks;
    int scaleout_rank_idx, scaleup_rank_idx;
    int num_sms, num_smem_bytes, num_channels;
    bool use_expanded_layout, allow_multiple_reduction;
};

struct HybridCombineReduceEpilogueArgs {
    void* combined_x; float* combined_topk_weights; topk_idx_t* combined_topk_idx;
    int num_combined_tokens, num_max_tokens_per_rank;
    int hidden, num_experts, num_topk;
    void* reduce_buffer;
    void* bias_0; void* bias_1;
    int num_scaleout_ranks, num_scaleup_ranks;
    int scaleout_rank_idx, scaleup_rank_idx;
    int num_sms, num_smem_bytes;
    bool use_expanded_layout, allow_multiple_reduction;
};

// Per-call inputs the default launchers do not need
struct HybridDispatchExtras {
    int* handle;
    bool do_expand, allow_multiple_reduction, prefer_overlap_with_compute;
};

struct HybridCombineExtras {
    int* handle;
    int num_combined_tokens;
};

struct HybridCombineReduceEpilogueExtras {
    int* handle;
    int num_channels;
};

class HybridKernelVariant {
public:
    virtual ~HybridKernelVariant() = default;

    virtual const char* name() const = 0;

    // GIN requirements of the device communicator. `num_allocated_qps` is the caller's
    // request with 0 meaning automatic; the returned `context_count` is the resolved QP count.
    virtual comm::GinRequirements get_gin_requirements(const int& num_allocated_qps,
                                                       const int& num_rdma_ranks) const = 0;

    // Scale-out buffer bytes (the scale-up receive buffer is shared and sized by the caller)
    virtual int64_t get_dispatch_scaleout_buffer_size(const layout::TokenLayout& token_layout,
                                                      const int& num_max_tokens_per_rank,
                                                      const int& num_scaleout_ranks,
                                                      const int& num_max_channels) const = 0;
    virtual int64_t get_combine_scaleout_buffer_size(const layout::TokenLayout& token_layout,
                                                     const int& num_max_tokens_per_rank,
                                                     const int& num_topk,
                                                     const int& num_scaleout_ranks,
                                                     const int& num_max_channels,
                                                     const bool& allow_multiple_reduction) const = 0;

    // Channel budget of one launch. Receives the default decision and may only lower it.
    virtual int get_num_channels_per_sm(const int& num_channels_per_sm,
                                        const int& num_sms, const int& num_qps,
                                        const int& num_smem_bytes, const int& num_notify_smem_bytes,
                                        const layout::TokenLayout& dispatch_token_layout,
                                        const layout::TokenLayout& combine_token_layout,
                                        const bool& prefer_overlap_with_compute) const = 0;

    // Optional per-dispatch handle tensor, written by dispatch and read by combine. The
    // Python `EPHandle` carries it without interpreting it.
    virtual bool has_dispatch_handle() const = 0;
    virtual torch::Tensor make_dispatch_handle(const int& num_max_tokens_per_rank,
                                              const int& num_topk) const = 0;
    virtual void check_dispatch_handle(const torch::Tensor& handle,
                                       const int& num_max_tokens_per_rank,
                                       const int& num_topk) const = 0;

    // Kernel launches. The combine launch returns the buffer the reduce epilogue reads.
    virtual void launch_dispatch(const HybridDispatchArgs& args, const HybridDispatchExtras& extras,
                                 const at::cuda::CUDAStream& stream) const = 0;
    virtual void* launch_combine(const HybridCombineArgs& args, const HybridCombineExtras& extras,
                                 const at::cuda::CUDAStream& stream) const = 0;
    virtual void launch_combine_reduce_epilogue(const HybridCombineReduceEpilogueArgs& args,
                                                const HybridCombineReduceEpilogueExtras& extras,
                                                const at::cuda::CUDAStream& stream) const = 0;
};

// Pick the variant for a communicator, or nullptr for the default kernels. Direct mode
// (`allow_hybrid_mode == false`) never uses a variant.
static std::shared_ptr<HybridKernelVariant> select_hybrid_kernel_variant(const int64_t& nccl_comm,
                                                                         const bool& allow_hybrid_mode) {
    return nullptr;
}

} // namespace deep_ep::ep
