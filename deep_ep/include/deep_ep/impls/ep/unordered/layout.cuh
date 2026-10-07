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

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/ep/token.cuh>

namespace deep_ep::ep::unordered {

// Per-channel shared-memory TMA buffers of the dispatch kernel: one for the scale-out send
// warp and two for the forward warp when its loads are double-buffered. The host channel
// budget and the kernel pool must agree on these.
constexpr int kNumDispatchSendBuffers = 1;
constexpr int kNumDispatchFwdBuffers = 2;
constexpr int kNumDispatchBuffersPerChannel = kNumDispatchSendBuffers + kNumDispatchFwdBuffers;

// A scale-out slot of the dispatch kernel: the token bytes as the TMA copies move them
// (`layout::TokenLayout` without header), followed by an 8-byte in-band header that
// describes the batch the slot starts. The header sits in the metadata padding when it
// fits there, so the slot is usually as large as the token itself.
struct ScaleoutSlotLayout {
    layout::TokenLayout token_layout;
    void* base;

    static constexpr int kHeaderBytes = sizeof(int64_t);

    __forceinline__ __device__ __host__
    explicit ScaleoutSlotLayout(const layout::TokenLayout& token_layout, void* base = nullptr) :
        token_layout(token_layout), base(base) {}

    __forceinline__ __device__ __host__ int get_metadata_offset() const {
        return math::align(token_layout.num_hidden_bytes, kNumTMAAlignmentBytes) +
               math::align(token_layout.num_sf_bytes, kNumTMAAlignmentBytes);
    }

    __forceinline__ __device__ __host__ int get_header_offset() const {
        return get_metadata_offset() + math::align<int>(token_layout.num_metadata_bytes, kHeaderBytes);
    }

    template <typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        const auto num_metadata_bytes = math::align<int>(token_layout.num_metadata_bytes, kHeaderBytes) + kHeaderBytes;
        return static_cast<dtype_t>(math::align<int>(
            get_metadata_offset() + math::align(num_metadata_bytes, kNumTMAAlignmentBytes), kNumRDMAAlignmentBytes));
    }

    __forceinline__ __device__ __host__ void* get_base_ptr() const {
        return base;
    }

    __forceinline__ __device__ __host__ int64_t* get_header_ptr() const {
        return math::advance_ptr<int64_t>(base, get_header_offset());
    }
};

// `layout::BufferLayout` over scale-out slots
struct ScaleoutBufferLayout {
    ScaleoutSlotLayout slot_layout;
    int num_ranks;
    int num_slots_per_rank;
    void* base;

    __forceinline__ __device__ __host__
    ScaleoutBufferLayout(const ScaleoutSlotLayout& slot_layout,
                         const int& num_ranks, const int& num_slots_per_rank,
                         void* base = nullptr) :
        slot_layout(slot_layout), num_ranks(num_ranks), num_slots_per_rank(num_slots_per_rank), base(base) {}

    __forceinline__ __device__ __host__ int64_t get_num_bytes_per_slot() const {
        return slot_layout.get_num_bytes<int64_t>();
    }

    __forceinline__ __device__ __host__ int64_t get_num_bytes_per_rank() const {
        return num_slots_per_rank * get_num_bytes_per_slot();
    }

    __forceinline__ __device__ __host__ int64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    __forceinline__ __device__ __host__ void* get_buffer_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    __forceinline__ __device__ __host__ ScaleoutBufferLayout get_rank_buffer(const int& rank_idx) const {
        return ScaleoutBufferLayout(slot_layout, 1, num_slots_per_rank,
                                    static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    template <int kNumSlotsPerChannel>
    __forceinline__ __device__ __host__ ScaleoutBufferLayout get_channel_buffer(const int& channel_idx) const {
        EP_UNIFIED_ASSERT(num_slots_per_rank % kNumSlotsPerChannel == 0);
        // Keep the rank stride, only the base moves
        return ScaleoutBufferLayout(slot_layout, num_ranks, num_slots_per_rank,
                                    static_cast<int8_t*>(base) + get_num_bytes_per_slot() * kNumSlotsPerChannel * channel_idx);
    }

    __forceinline__ __device__ __host__ ScaleoutSlotLayout get_slot(const int& slot_idx) const {
        EP_UNIFIED_ASSERT(num_ranks == 1);
        return ScaleoutSlotLayout(slot_layout.token_layout,
                                  static_cast<int8_t*>(base) + get_num_bytes_per_slot() * slot_idx);
    }
};

// Packing of the per-(token, k) receive map that dispatch records for the combine
// epilogue: where the partial for expert k of token T lands in the scale-out receive
// buffer. Layout (int32):
//   bits [30:26]  destination scale-out rank (5 bits)
//   bits [25:12]  slot within `recv[rank][channel * slots_per_channel + slot]` (14 bits)
//   bits [11:0]   channel index (12 bits)
// The fields add up to 31 bits, so bit 31 stays clear on a valid entry and the epilogue can
// keep the `< 0` sentinel of `compute_topk_slots` for unused top-k entries.
constexpr int kCombineRecvMapChannelBits = 12;
constexpr int kCombineRecvMapSlotBits = 14;
constexpr int kCombineRecvMapRankBits = 5;
constexpr int kCombineRecvMapChannelMask = (1 << kCombineRecvMapChannelBits) - 1;
constexpr int kCombineRecvMapSlotMask = (1 << kCombineRecvMapSlotBits) - 1;
constexpr int kCombineRecvMapRankMask = (1 << kCombineRecvMapRankBits) - 1;
constexpr int kCombineRecvMapSlotShift = kCombineRecvMapChannelBits;
constexpr int kCombineRecvMapRankShift = kCombineRecvMapChannelBits + kCombineRecvMapSlotBits;

__device__ __host__ __forceinline__
int pack_combine_recv_addr(const int& rank, const int& slot, const int& channel) {
    return (rank << kCombineRecvMapRankShift) | (slot << kCombineRecvMapSlotShift) | channel;
}

__device__ __host__ __forceinline__
void unpack_combine_recv_addr(const int& packed, int& rank, int& slot, int& channel) {
    rank = (packed >> kCombineRecvMapRankShift) & kCombineRecvMapRankMask;
    slot = (packed >> kCombineRecvMapSlotShift) & kCombineRecvMapSlotMask;
    channel = packed & kCombineRecvMapChannelMask;
}

}  // namespace deep_ep::ep::unordered
