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

#include <deep_ep/common/exception.cuh>

namespace deep_ep::ep::unordered {

// CTA-scope acquire/release accesses for the proxy hand-off rings in shared memory
template <typename dtype_t>
__forceinline__ __device__ dtype_t ld_acquire_cta(const dtype_t* ptr) {
    if constexpr (sizeof(dtype_t) == 4) {
        uint32_t value;
        asm volatile("ld.acquire.cta.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
        return reinterpret_cast<const dtype_t&>(value);
    } else if constexpr (sizeof(dtype_t) == 8) {
        uint64_t value;
        asm volatile("ld.acquire.cta.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
        return reinterpret_cast<const dtype_t&>(value);
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_release_cta(void* ptr, dtype_t value) {
    if constexpr (sizeof(dtype_t) == 4) {
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);
        asm volatile("st.release.cta.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));
    } else if constexpr (sizeof(dtype_t) == 8) {
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);
        asm volatile("st.release.cta.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
    }
}

// Explicit register handoff between the scale-up and forward warps of the combine kernel.
// Unlike `ptx::warpgroup_reg_realloc`, the direction is chosen by the caller: the forward
// warps grow and the scale-up warps shrink regardless of the launch-bounds budget.
template <int kNumRegs>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
}

template <int kNumRegs>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
}

}  // namespace deep_ep::ep::unordered
