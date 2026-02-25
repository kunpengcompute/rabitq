/*
   Copyright 2026 Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#ifdef __aarch64__
    #include <arm_neon.h>
#ifndef float16_t
    typedef __fp16 float16_t;
#endif
#else
    #include <immintrin.h>
#endif
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

#include "matrix.h"
#include "utils.h"
#include <random>

#if defined(__GNUC__) && defined(__aarch64__)

#define KRL_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define KRL_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#else
#define KRL_IMPRECISE_FUNCTION_BEGIN
#define KRL_IMPRECISE_FUNCTION_END
#endif

template <uint32_t D, uint32_t B>
class Space {
public:
    // ================================================================================================
    // ********************
    //   Binary Operation
    // ********************
    inline static uint32_t popcount(u_int64_t *d);
    inline static uint32_t ip_bin_bin(uint64_t * q, uint64_t * d);
    inline static uint32_t ip_byte_bin(uint64_t *q, uint64_t *d);
    inline static void     transpose_bin(uint8_t *q, uint64_t *tq);
    
    // ================================================================================================
    inline static void range(float* q, float* c, float &vl, float &vr);
    inline static void  quantize(uint8_t *result, float *q, float* c, float *u, float max_entry, float width, uint32_t & sum_q);
    inline static uint32_t sum(uint8_t* d);
    Space(){};
    ~Space(){};
};

#ifdef __aarch64__
KRL_IMPRECISE_FUNCTION_BEGIN
void quant_f16(const float* src, size_t n, float16_t* out) {
    if (src == nullptr || out == nullptr || n == 0) {
        return;
    }
    size_t l = 0;
    constexpr size_t single_loop = 4;
    constexpr size_t multi_loop = 16;
    for (; l + multi_loop <= n; l += multi_loop) {
        const float32x4_t neon_a1 = vld1q_f32(src + l);
        const float32x4_t neon_a2 = vld1q_f32(src + l + 4);
        const float32x4_t neon_a3 = vld1q_f32(src + l + 8);
        const float32x4_t neon_a4 = vld1q_f32(src + l + 12);
        const float16x4_t neon_c1 = vcvt_f16_f32(neon_a1);
        const float16x4_t neon_c2 = vcvt_f16_f32(neon_a2);
        const float16x4_t neon_c3 = vcvt_f16_f32(neon_a3);
        const float16x4_t neon_c4 = vcvt_f16_f32(neon_a4);
        vst1_f16(out + l, neon_c1);
        vst1_f16(out + l + 4, neon_c2);
        vst1_f16(out + l + 8, neon_c3);
        vst1_f16(out + l + 12, neon_c4);
    } 
    for (; l + single_loop <= n; l += single_loop) {
        const float32x4_t neon_a1 = vld1q_f32(src + l);
        const float16x4_t neon_c1 = vcvt_f16_f32(neon_a1);
        vst1_f16(out + l, neon_c1);
    }
    for (; l < n; ++l) {
        out[l] = (float16_t)(src[l]);
    }
}
KRL_IMPRECISE_FUNCTION_END

// NEON movemask模拟实现
inline uint32_t neon_movemask(uint8x16_t input) {
    // 将每个字节的最高位提取到bit0
    const uint8x16_t mask = vdupq_n_u8(0x80);
    uint8x16_t high_bits = vandq_u8(input, mask);
    
    // 将符号位移动到最低位并转换为位掩码
    high_bits = vshrq_n_u8(high_bits, 7); // 现在每个元素是0x00或0x01
    
    // 将8位值压缩到64位
    uint64x2_t tmp = vreinterpretq_u64_u8(high_bits);
    uint64_t lo = vgetq_lane_u64(tmp, 0);
    uint64_t hi = vgetq_lane_u64(tmp, 1);
    
    // 使用位操作生成掩码
    lo = (lo & 0x0101010101010101) * 0x0102040810204080;
    hi = (hi & 0x0101010101010101) * 0x0102040810204080;
    return ((hi >> 56) << 8) | (lo >> 56);
}

// ==============================================================
// decompose the quantized query vector into B_q binary vector
// ==============================================================
template <uint32_t D, uint32_t B>
void Space<D, B>::transpose_bin(uint8_t *q, uint64_t *tq) {
    if (q == nullptr || tq == nullptr) {
        return;
    }
    for (int i = 0; i < B; i += 32) {
        // 加载32字节到两个NEON寄存器
        uint8x16_t v_low = vld1q_u8(q);
        uint8x16_t v_high = vld1q_u8(q + 16);

        // 初始左移：每个32位元素左移(8 - B_QUERY)位
        uint32x4_t vl = vshlq_n_u32(vreinterpretq_u32_u8(v_low), (8 - B_QUERY));
        uint32x4_t vh = vshlq_n_u32(vreinterpretq_u32_u8(v_high), (8 - B_QUERY));

        for (int j = 0; j < B_QUERY; j++) {
            // 将向量转换回字节类型并提取掩码
            uint8x16_t bl = vreinterpretq_u8_u32(vl);
            uint8x16_t bh = vreinterpretq_u8_u32(vh);

            // 提取每个字节的最高位生成32位掩码
            uint32_t mask_low = neon_movemask(bl);
            uint32_t mask_high = neon_movemask(bh);
            uint32_t v1 = (mask_high << 16) | mask_low;

            v1 = reverseBits(v1); // 使用原reverseBits函数

            // 计算存储位置
            int idx = (B_QUERY - j - 1) * (B / 64) + i / 64;
            // uint64_t shift = (i / 32 % 2 == 0) ? 32 : 0;
            uint64_t shift = ((i / 32) & 1) ? 0 : 32;

            tq[idx] |= ((uint64_t)v1 << shift);

            // 左移每个32位元素1位
            vl = vshlq_n_u32(vl, 1);
            vh = vshlq_n_u32(vh, 1);
        }
        q += 32;
    }
}
#else
// ==============================================================
// decompose the quantized query vector into B_q binary vector
// ==============================================================
template <uint32_t D, uint32_t B>
void Space<D, B>::transpose_bin(uint8_t *q, uint64_t *tq){
    if (q == nullptr || tq == nullptr) {
        return;
    }
    for(int i=0;i<B;i+=32){
        __m256i v = _mm256_load_si256(reinterpret_cast<__m256i*>(q));
        v = _mm256_slli_epi32(v, (8-B_QUERY));
        for(int j=0;j<B_QUERY;j++){
            uint32_t v1 = _mm256_movemask_epi8(v);
            v1 = reverseBits(v1);
            tq[(B_QUERY - j - 1) * (B / 64) + i / 64] |= ((uint64_t)v1 << ((i / 32 % 2 == 0) ? 32:0));
            v = _mm256_slli_epi32(v, 1);
        }
        q += 32;
    }
}
#endif

// ==============================================================
// inner product between binary strings
// ==============================================================
template <uint32_t D, uint32_t B>
inline uint32_t Space<D, B>::ip_bin_bin(uint64_t * q, uint64_t * d){
    uint64_t ret = 0;
    for(int i = 0; i < B / 64; i ++){
        ret += __builtin_popcountll((*d) & (*q));
        q ++;
        d ++;
    }
    return ret;
}

// ==============================================================
// popcount (a.k.a, bitcount)
// ==============================================================
template <uint32_t D, uint32_t B>
inline uint32_t Space<D, B>::popcount(u_int64_t *d){
    uint64_t ret = 0;
    for (int i = 0; i < B / 64; i++) {
        ret += __builtin_popcountll((*d));
        d++;
    }
    return ret;
}

// ==============================================================
// inner product between a decomposed byte string q
// and a binary string d
// ==============================================================
template <uint32_t D, uint32_t B>
uint32_t Space<D, B>::ip_byte_bin(uint64_t *q, uint64_t *d){
    uint64_t ret = 0;
    for(int i = 0; i < B_QUERY; i++){
        ret += (ip_bin_bin(q, d) << i);
        q += (B / 64);
    }
    return ret;
}

// ==============================================================
// compute the min and max value of the entries of q
// ==============================================================
template <uint32_t D, uint32_t B>
void Space<D, B>::range(float *q, float *c, float &vl, float &vr)
{
    vl = +1e20;
    vr = -1e20;
    #pragma clang loop vectorize(assume_safety)
    for (int i = 0; i < B; i++) {
        float tmp = (*q) - (*c);
        vl = std::min(tmp, vl);
        vr = std::max(vr, tmp);
        q++;
        c++;
    }
}

// ==============================================================
// quantize the query vector with uniform scalar quantization
// ==============================================================
template <uint32_t D, uint32_t B>
void Space<D, B>::quantize(uint8_t *result, float *q, float *c, float *u, float vl, float width, uint32_t &sum_q)
{
    float one_over_width = 1.0 / width;
    uint8_t *ptr_res = result;
    uint32_t sum = 0;
    for (int i = 0; i < B; i++) {
        (*ptr_res) = (uint8_t)((((*q) - (*c)) - vl) * one_over_width + (*u)); // + (*u)
        sum += (*ptr_res);
        q++;
        c++;
        ptr_res++;
        u++;
    }
    sum_q = sum;
}

#ifdef __aarch64__
#define PD2 3 
template <uint32_t d>
inline float krl_L2sqr_f16f32(const float16_t* x, const float16_t* __restrict y) {
    uint32_t i = 0;
    constexpr size_t single_round = 8;
    constexpr size_t double_round = 32;

    /* Initialize result registers */
    float32x4_t res1 = vdupq_n_f32(0.0f);
    float32x4_t res2 = vdupq_n_f32(0.0f);
    float32x4_t res3 = vdupq_n_f32(0.0f);
    float32x4_t res4 = vdupq_n_f32(0.0f);

    /* Main computation loop with double rounds */
    for (; i + double_round <= d; i += double_round) {
        __builtin_prefetch(y + i + double_round * PD2, 0, 3);
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t x8_1 = vld1q_f16(x + i + 8);

        float16x8_t y8_0 = vld1q_f16(y + i);
        float16x8_t y8_1 = vld1q_f16(y + i + 8);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0); 
        float16x8_t d8_1 = vsubq_f16(x8_1, y8_1);

        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_low_f16(res2, d8_1, d8_1);

        res1 = vfmlalq_high_f16(res1, d8_0, d8_0);
        res2 = vfmlalq_high_f16(res2, d8_1, d8_1);

        __builtin_prefetch(y + i + double_round * PD2 + 16, 0, 3);
        float16x8_t x8_2 = vld1q_f16(x + i + 16);
        float16x8_t x8_3 = vld1q_f16(x + i + 24);
        float16x8_t y8_2 = vld1q_f16(y + i + 16);
        float16x8_t y8_3 = vld1q_f16(y + i + 24);
        float16x8_t d8_2 = vsubq_f16(x8_2, y8_2);
        float16x8_t d8_3 = vsubq_f16(x8_3, y8_3);
        res3 = vfmlalq_low_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_low_f16(res4, d8_3, d8_3);
        res3 = vfmlalq_high_f16(res3, d8_2, d8_2);
        res4 = vfmlalq_high_f16(res4, d8_3, d8_3);
    }

    if constexpr ((d & 31) == 0) {
        res1 = vaddq_f32(res1, res2);
        res3 = vaddq_f32(res3, res4);
        res1 = vaddq_f32(res1, res3);
        return vaddvq_f32(res1);
    }

    /* Handle remaining elements with single rounds */
    for (; i + single_round <= d; i += single_round) {
        float16x8_t x8_0 = vld1q_f16(x + i);
        float16x8_t y8_0 = vld1q_f16(y + i);

        float16x8_t d8_0 = vsubq_f16(x8_0, y8_0); 
        res1 = vfmlalq_low_f16(res1, d8_0, d8_0);
        res3 = vfmlalq_high_f16(res3, d8_0, d8_0);
    }
    /* Accumulate results */
    res1 = vaddq_f32(res1, res2);
    res3 = vaddq_f32(res3, res4);
    res1 = vaddq_f32(res1, res3);
    float res = vaddvq_f32(res1);
    /* Handle remaining elements */
    if constexpr (d & 7) {
        for (; i < d; i++) {
            const float16_t tmp = x[i] - y[i];
            res += ((float)tmp * (float)tmp);
        }
    }
    return res;
}

template <uint32_t d>
inline float krl_L2sqr_f32f16f32(const float* x, const float16_t* y) {
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 16;
    size_t i;
    float res;

    if constexpr(d >= multi_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t x8_1 = vld1q_f32(x + 4);
        float32x4_t x8_2 = vld1q_f32(x + 8);
        float32x4_t x8_3 = vld1q_f32(x + 12);

        float32x4_t y8_0 = vcvt_f32_f16(vld1_f16(y));
        float32x4_t y8_1 = vcvt_f32_f16(vld1_f16(y + 4));
        float32x4_t y8_2 = vcvt_f32_f16(vld1_f16(y + 8));
        float32x4_t y8_3 = vcvt_f32_f16(vld1_f16(y + 12));

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        float32x4_t d8_1 = vsubq_f32(x8_1, y8_1);
        d8_1 = vmulq_f32(d8_1, d8_1);
        float32x4_t d8_2 = vsubq_f32(x8_2, y8_2);
        d8_2 = vmulq_f32(d8_2, d8_2);
        float32x4_t d8_3 = vsubq_f32(x8_3, y8_3);
        d8_3 = vmulq_f32(d8_3, d8_3);

        for (i = multi_round; i <= d - multi_round; i += multi_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vcvt_f32_f16(vld1_f16(y + i));
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);

            x8_1 = vld1q_f32(x + i + 4);
            y8_1 = vcvt_f32_f16(vld1_f16(y + i + 4));
            const float32x4_t q8_1 = vsubq_f32(x8_1, y8_1);
            d8_1 = vmlaq_f32(d8_1, q8_1, q8_1);

            x8_2 = vld1q_f32(x + i + 8);
            y8_2 = vcvt_f32_f16(vld1_f16(y + i + 8));
            const float32x4_t q8_2 = vsubq_f32(x8_2, y8_2);
            d8_2 = vmlaq_f32(d8_2, q8_2, q8_2);

            x8_3 = vld1q_f32(x + i + 12);
            y8_3 = vcvt_f32_f16(vld1_f16(y + i + 12));
            const float32x4_t q8_3 = vsubq_f32(x8_3, y8_3);
            d8_3 = vmlaq_f32(d8_3, q8_3, q8_3);
        }

        for(; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vcvt_f32_f16(vld1_f16(y + i));
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }
        
        d8_0 = vaddq_f32(d8_0, d8_1);
        d8_2 = vaddq_f32(d8_2, d8_3);
        d8_0 = vaddq_f32(d8_0, d8_2);
        res = vaddvq_f32(d8_0);
    } else if constexpr(d >= single_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t y8_0 = vcvt_f32_f16(vld1_f16(y));

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        for(i = single_round; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vcvt_f32_f16(vld1_f16(y + i));
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }
        res = vaddvq_f32(d8_0);
    } else {
        res = 0;
        i = 0;
    }
    if constexpr(d & 3) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
        for (i++; i < d; i++) {
            tmp = x[i] - (float)y[i];
            res += tmp * tmp;
        }
    }
    return res;
}

template <uint32_t d>
inline float sqr_dist(const float* x, const float* y) {
    if constexpr(d == 784) {
        float sum ;
        for (uint32_t i = 0; i < d; ++i) {
            float tmp = x[i] - y[i];
            sum += tmp * tmp;
        }
        return sum;
    }
    constexpr size_t single_round = 4;
    constexpr size_t multi_round = 16;
    size_t i;
    float res;

    if constexpr(d >= multi_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t x8_1 = vld1q_f32(x + 4);
        float32x4_t x8_2 = vld1q_f32(x + 8);
        float32x4_t x8_3 = vld1q_f32(x + 12);

        float32x4_t y8_0 = vld1q_f32(y);
        float32x4_t y8_1 = vld1q_f32(y + 4);
        float32x4_t y8_2 = vld1q_f32(y + 8);
        float32x4_t y8_3 = vld1q_f32(y + 12);

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        float32x4_t d8_1 = vsubq_f32(x8_1, y8_1);
        d8_1 = vmulq_f32(d8_1, d8_1);
        float32x4_t d8_2 = vsubq_f32(x8_2, y8_2);
        d8_2 = vmulq_f32(d8_2, d8_2);
        float32x4_t d8_3 = vsubq_f32(x8_3, y8_3);
        d8_3 = vmulq_f32(d8_3, d8_3);

        for (i = multi_round; i <= d - multi_round; i += multi_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);

            x8_1 = vld1q_f32(x + i + 4);
            y8_1 = vld1q_f32(y + i + 4);
            const float32x4_t q8_1 = vsubq_f32(x8_1, y8_1);
            d8_1 = vmlaq_f32(d8_1, q8_1, q8_1);

            x8_2 = vld1q_f32(x + i + 8);
            y8_2 = vld1q_f32(y + i + 8);
            const float32x4_t q8_2 = vsubq_f32(x8_2, y8_2);
            d8_2 = vmlaq_f32(d8_2, q8_2, q8_2);

            x8_3 = vld1q_f32(x + i + 12);
            y8_3 = vld1q_f32(y + i + 12);
            const float32x4_t q8_3 = vsubq_f32(x8_3, y8_3);
            d8_3 = vmlaq_f32(d8_3, q8_3, q8_3);
        }

        for(; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }
        
        d8_0 = vaddq_f32(d8_0, d8_1);
        d8_2 = vaddq_f32(d8_2, d8_3);
        d8_0 = vaddq_f32(d8_0, d8_2);
        res = vaddvq_f32(d8_0);
    } else if constexpr(d >= single_round) {
        float32x4_t x8_0 = vld1q_f32(x);
        float32x4_t y8_0 = vld1q_f32(y);

        float32x4_t d8_0 = vsubq_f32(x8_0, y8_0);
        d8_0 = vmulq_f32(d8_0, d8_0);
        for(i = single_round; i <= d - single_round; i += single_round) {
            x8_0 = vld1q_f32(x + i);
            y8_0 = vld1q_f32(y + i);
            const float32x4_t q8_0 = vsubq_f32(x8_0, y8_0);
            d8_0 = vmlaq_f32(d8_0, q8_0, q8_0);
        }
        res = vaddvq_f32(d8_0);
    } else {
        res = 0;
        i = 0;
    }
    if constexpr(d & 3) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
        for (i++; i < d; i++) {
            tmp = x[i] - y[i];
            res += tmp * tmp;
        }
    }
    return res;
}

template <>
float sqr_dist<128>(const float *d, const float *q) {
    float32x4_t sum0={0}, sum1={0}, sum2={0}, sum3={0};
    constexpr uint32_t unroll = 8;
    uint32_t i=0;
    for(; i+4*unroll<=128; i+=4*unroll) {
        for(uint32_t j=0; j<unroll; ++j) {
            float32x4_t dx = vld1q_f32(d+i+j*4);
            float32x4_t qx = vld1q_f32(q+i+j*4);
            float32x4_t diff = vsubq_f32(dx, qx);

            // 轮换使用累加器
          switch(j&3) {
              case 0: sum0 = vmlaq_f32(sum0, diff, diff); break;
              case 1: sum1 = vmlaq_f32(sum1, diff, diff); break;
              case 2: sum2 = vmlaq_f32(sum2, diff, diff); break;
              case 3: sum3 = vmlaq_f32(sum3, diff, diff); break;
          }
        }
    }

    sum0 = vaddq_f32(vaddq_f32(sum0,sum1), vaddq_f32(sum2,sum3));
    float sum = vaddvq_f32(sum0);
    return sum;
}
#else
// The implementation is based on https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_ip.h
template<uint32_t L>
inline float sqr_dist(float *d, float *q) {
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr uint32_t num_blk16 = L >> 4;
    constexpr uint32_t l = L & 0b1111;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for(int i=0;i<num_blk16;i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
}
    for(int i=0;i<l/8;i++){
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    
    for(int i=0;i<l%8;i++){
        float tmp = (*q) - (*d);
        ret += tmp * tmp;
        d ++;
        q ++;
    }
    return ret;
}
#endif