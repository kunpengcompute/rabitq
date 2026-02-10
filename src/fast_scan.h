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

// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)
#pragma once
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <assert.h>
#ifdef __aarch64__
    #include <arm_neon.h>
#else
    #include <immintrin.h>
#endif

#define lowbit(x) (x&(-x))
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

using namespace std;

#ifdef __aarch64__
extern "C" {
    void krl_table_lookup_fast_scan_bs64_asm(
        int nsq, 
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* distance);
}

template <uint32_t B>
inline __attribute__((always_inline)) void accumulate_one_block_32(uint8_t* __restrict codes,
                                 uint8_t* __restrict LUT,
                                 uint16_t* __restrict distance) {
    constexpr int roundup_B = (B + 3) / 4;
    uint16x8_t result[8] = {vdupq_n_u16(0)};
    const uint8x16_t mask16 = vreinterpretq_u8_u16(vdupq_n_u16(0x1000));
    const uint8x16_t mask01 = vreinterpretq_u8_u16(vdupq_n_u16(0x0100));
    /* main loop */
    #pragma unroll
    for (int sq = 0; sq < roundup_B; sq += 4) { 
        uint8x16_t mask0 = vld1q_u8(codes);
        uint8x16_t mask1 = vld1q_u8(codes + 16);
        uint8x16_t mask2 = vld1q_u8(codes + 32);
        uint8x16_t mask3 = vld1q_u8(codes + 48);
        codes += 64;
        uint8x16_t mask0_1 = vsliq_n_u8(mask0, mask01, 4);
        uint8x16_t mask0_2 = vsriq_n_u8(mask16, mask0, 4);
        uint8x16_t mask1_1 = vsliq_n_u8(mask1, mask01, 4);
        uint8x16_t mask1_2 = vsriq_n_u8(mask16, mask1, 4);
        uint8x16_t mask2_1 = vsliq_n_u8(mask2, mask01, 4);
        uint8x16_t mask2_2 = vsriq_n_u8(mask16, mask2, 4);
        uint8x16_t mask3_1 = vsliq_n_u8(mask3, mask01, 4);
        uint8x16_t mask3_2 = vsriq_n_u8(mask16, mask3, 4);
        uint8x16x2_t dictCombine0 = vld1q_u8_x2(LUT);
        uint8x16x2_t dictCombine1 = vld1q_u8_x2(LUT + 32);
        LUT += 64;
        
        mask0_1 = vqtbl2q_u8(dictCombine0, mask0_1);
        mask0_2 = vqtbl2q_u8(dictCombine0, mask0_2);
        mask1_1 = vqtbl2q_u8(dictCombine0, mask1_1);
        mask1_2 = vqtbl2q_u8(dictCombine0, mask1_2);
        mask2_1 = vqtbl2q_u8(dictCombine1, mask2_1);
        mask2_2 = vqtbl2q_u8(dictCombine1, mask2_2);
        mask3_1 = vqtbl2q_u8(dictCombine1, mask3_1);
        mask3_2 = vqtbl2q_u8(dictCombine1, mask3_2);

        result[0] = vpadalq_u8(result[0], mask0_1);
        result[1] = vpadalq_u8(result[1], mask0_2);
        result[2] = vpadalq_u8(result[2], mask1_1);
        result[3] = vpadalq_u8(result[3], mask1_2);
        result[4] = vpadalq_u8(result[4], mask2_1);
        result[5] = vpadalq_u8(result[5], mask2_2);
        result[6] = vpadalq_u8(result[6], mask3_1);
        result[7] = vpadalq_u8(result[7], mask3_2);
    }
    result[0] = vaddq_u16(result[0], result[4]);
    result[1] = vaddq_u16(result[1], result[5]);
    result[2] = vaddq_u16(result[2], result[6]);
    result[3] = vaddq_u16(result[3], result[7]);
    vst1q_u16(distance     , result[0]);
    vst1q_u16(distance + 8 , result[1]);
    vst1q_u16(distance + 16, result[2]);
    vst1q_u16(distance + 24, result[3]);
}

#define DUP_ZEROS96()               \
    result[0 ] = vdupq_n_u16(0);    \
    result[1 ] = vdupq_n_u16(0);    \
    result[2 ] = vdupq_n_u16(0);    \
    result[3 ] = vdupq_n_u16(0);    \
    result[4 ] = vdupq_n_u16(0);    \
    result[5 ] = vdupq_n_u16(0);    \
    result[6 ] = vdupq_n_u16(0);    \
    result[7 ] = vdupq_n_u16(0);    \
    result[8 ] = vdupq_n_u16(0);    \
    result[9 ] = vdupq_n_u16(0);    \
    result[10] = vdupq_n_u16(0);    \
    result[11] = vdupq_n_u16(0);

#define LOAD_TABLE()                \
    tables = vld1q_u8_x2(LUT);      \
    LUT += 32;

#define LOAD_CODES96()              \
    mask0_1 = vld1q_u8(codes);      \
    mask1_1 = vld1q_u8(codes + 16); \
    mask2_1 = vld1q_u8(codes + 32); \
    mask3_1 = vld1q_u8(codes + 48); \
    mask4_1 = vld1q_u8(codes + 64); \
    mask5_1 = vld1q_u8(codes + 80); \
    codes += 96;

#define SRI_SLI96()                             \
    mask0_2 = vsriq_n_u8(mask16, mask0_1, 4);   \
    mask1_2 = vsriq_n_u8(mask16, mask1_1, 4);   \
    mask2_2 = vsriq_n_u8(mask16, mask2_1, 4);   \
    mask3_2 = vsriq_n_u8(mask16, mask3_1, 4);   \
    mask4_2 = vsriq_n_u8(mask16, mask4_1, 4);   \
    mask5_2 = vsriq_n_u8(mask16, mask5_1, 4);   \
    mask0_1 = vsliq_n_u8(mask0_1, mask01, 4);   \
    mask1_1 = vsliq_n_u8(mask1_1, mask01, 4);   \
    mask2_1 = vsliq_n_u8(mask2_1, mask01, 4);   \
    mask3_1 = vsliq_n_u8(mask3_1, mask01, 4);   \
    mask4_1 = vsliq_n_u8(mask4_1, mask01, 4);   \
    mask5_1 = vsliq_n_u8(mask5_1, mask01, 4);   

#define TBL96()                             \
    mask0_1 = vqtbl2q_u8(tables, mask0_1);  \
    mask0_2 = vqtbl2q_u8(tables, mask0_2);  \
    mask1_1 = vqtbl2q_u8(tables, mask1_1);  \
    mask1_2 = vqtbl2q_u8(tables, mask1_2);  \
    mask2_1 = vqtbl2q_u8(tables, mask2_1);  \
    mask2_2 = vqtbl2q_u8(tables, mask2_2);  \
    mask3_1 = vqtbl2q_u8(tables, mask3_1);  \
    mask3_2 = vqtbl2q_u8(tables, mask3_2);  \
    mask4_1 = vqtbl2q_u8(tables, mask4_1);  \
    mask4_2 = vqtbl2q_u8(tables, mask4_2);  \
    mask5_1 = vqtbl2q_u8(tables, mask5_1);  \
    mask5_2 = vqtbl2q_u8(tables, mask5_2);  

#define PADDLQ96_0()                    \
    result[0 ] = vpaddlq_u8(mask0_1);   \
    result[2 ] = vpaddlq_u8(mask1_1);   \
    result[4 ] = vpaddlq_u8(mask2_1);   \
    result[6 ] = vpaddlq_u8(mask3_1);   \
    result[8 ] = vpaddlq_u8(mask4_1);   \
    result[10] = vpaddlq_u8(mask5_1);   
            
#define PADDLQ96_1()                    \
    result[1 ] = vpaddlq_u8(mask0_2);   \
    result[3 ] = vpaddlq_u8(mask1_2);   \
    result[5 ] = vpaddlq_u8(mask2_2);   \
    result[7 ] = vpaddlq_u8(mask3_2);   \
    result[9 ] = vpaddlq_u8(mask4_2);   \
    result[11] = vpaddlq_u8(mask5_2);   

#define UADALP96_0()                                \
    result[0 ] = vpadalq_u8(result[0 ], mask0_1);   \
    result[2 ] = vpadalq_u8(result[2 ], mask1_1);   \
    result[4 ] = vpadalq_u8(result[4 ], mask2_1);   \
    result[6 ] = vpadalq_u8(result[6 ], mask3_1);   \
    result[8 ] = vpadalq_u8(result[8 ], mask4_1);   \
    result[10] = vpadalq_u8(result[10], mask5_1);   
            
#define UADALP96_1()                                \
    result[1 ] = vpadalq_u8(result[1 ], mask0_2);   \
    result[3 ] = vpadalq_u8(result[3 ], mask1_2);   \
    result[5 ] = vpadalq_u8(result[5 ], mask2_2);   \
    result[7 ] = vpadalq_u8(result[7 ], mask3_2);   \
    result[9 ] = vpadalq_u8(result[9 ], mask4_2);   \
    result[11] = vpadalq_u8(result[11], mask5_2);   

#define PD1 3

template <uint32_t B>
inline __attribute__((always_inline)) void accumulate_one_block_96(const uint8_t* __restrict codes,
                          const uint8_t* __restrict LUT,
                          uint16_t* __restrict distance) {
    constexpr int roundup_B = (B + 3) / 4;
    constexpr int nsq_sub2 = roundup_B + (roundup_B % 2) - 2;
    uint16x8_t result[12];
    uint8x16_t mask0_1, mask1_1, mask2_1, mask3_1, mask4_1, mask5_1;
    uint8x16_t mask0_2, mask1_2, mask2_2, mask3_2, mask4_2, mask5_2;
    uint8x16x2_t tables;
    LOAD_CODES96()
    const uint8x16_t mask16 = vreinterpretq_u8_u16(vdupq_n_u16(0x1000));
    const uint8x16_t mask01 = vreinterpretq_u8_u16(vdupq_n_u16(0x0100));
    LOAD_TABLE()
    if constexpr (nsq_sub2 > 0) {
        SRI_SLI96()
        TBL96()
        LOAD_TABLE()
        PADDLQ96_0()
        LOAD_CODES96()
        PADDLQ96_1()
    } else {
        DUP_ZEROS96()
    }
    for (int sq = 2; sq < nsq_sub2; sq += 2) { 
        __builtin_prefetch(LUT + PD1 * 32, 0, 3);
        __builtin_prefetch(codes + PD1 * 96, 0, 0);
        __builtin_prefetch(codes + PD1 * 96 + 64, 0, 0);
        SRI_SLI96()
        TBL96()
        UADALP96_0()
        LOAD_CODES96()
        UADALP96_1()
        LOAD_TABLE()
    }
    {
        SRI_SLI96()
        TBL96()
        UADALP96_0()
        UADALP96_1()
    }
    
    vst1q_u16(distance     , result[0 ]);
    vst1q_u16(distance + 8 , result[1 ]);
    vst1q_u16(distance + 16, result[2 ]);
    vst1q_u16(distance + 24, result[3 ]);
    vst1q_u16(distance + 32, result[4 ]);
    vst1q_u16(distance + 40, result[5 ]);
    vst1q_u16(distance + 48, result[6 ]);
    vst1q_u16(distance + 56, result[7 ]);
    vst1q_u16(distance + 64, result[8 ]);
    vst1q_u16(distance + 72, result[9 ]);
    vst1q_u16(distance + 80, result[10]);
    vst1q_u16(distance + 88, result[11]);
}

// ==============================================================
// look up the tables for all the packed batches
// ==============================================================
template <uint32_t B, uint32_t bbs = 32>
inline void accumulate(uint8_t* codes, uint8_t* LUT, uint16_t* result) {
    if constexpr (bbs == 96) {
        accumulate_one_block_96<B>(codes, LUT, result);
    } else if constexpr (bbs == 64) {
        krl_table_lookup_fast_scan_bs64_asm((B + 3) / 4, codes, LUT, result);
    } else if constexpr (bbs == 32) {
        accumulate_one_block_32<B>(codes, LUT, result);
    }
}
#else
// ==============================================================
// look up the tables for a packed batch of 32 quantization codes
// ==============================================================
template <uint32_t B>
inline void accumulate_one_block(uint8_t* codes, uint8_t* LUT, uint16_t* result){
    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu[4];
    for(int i=0;i<4;i++){
        accu[i] = _mm256_setzero_si256();   
}

    constexpr uint32_t M = B / 4;

    for(int m=0;m<M;m+=2){
        __m256i c   = _mm256_load_si256((__m256i const*)codes);
        __m256i lo  = _mm256_and_si256(c, low_mask);
        __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        __m256i lut = _mm256_load_si256((__m256i const*)LUT);

        __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
        __m256i res_hi = _mm256_shuffle_epi8(lut, hi);

        accu[0] = _mm256_add_epi16(accu[0], res_lo);
        accu[1] = _mm256_add_epi16(accu[1], _mm256_srli_epi16(res_lo, 8));

        accu[2] = _mm256_add_epi16(accu[2], res_hi);
        accu[3] = _mm256_add_epi16(accu[3], _mm256_srli_epi16(res_hi, 8));
        
        codes += 32;
        LUT   += 32;
    }

    accu[0] = _mm256_sub_epi16(accu[0], _mm256_slli_epi16(accu[1], 8));
    __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[0], accu[1], 0x21),_mm256_blend_epi32(accu[0], accu[1], 0xF0));
    _mm256_store_si256((__m256i*)(result + 0 ), dis0);
    
    accu[2] = _mm256_sub_epi16(accu[2], _mm256_slli_epi16(accu[3], 8));
    __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[2], accu[3], 0x21),_mm256_blend_epi32(accu[2], accu[3], 0xF0));
    _mm256_store_si256((__m256i*)(result + 16), dis1);    
}

// ==============================================================
// look up the tables for all the packed batches
// ==============================================================
template <uint32_t B>
inline void accumulate(uint32_t nblk, uint8_t* codes, uint8_t* LUT, uint16_t* result){
    for(int i=0;i<nblk;i++){
        accumulate_one_block<B>(codes, LUT, result);
        codes  += 32 * B / 8;
        result += 32;
    }
}
#endif

// ==============================================================
// prepare the look-up-table from the quantized query vector
// ==============================================================
#ifdef __aarch64__
template <uint32_t B>
inline void pack_LUT(uint8_t* byte_query, uint8_t* LUT){
    constexpr uint32_t M = B / 64;
    // The NEON intrinsics require 16 indices, remaining are ignored.
    // // --- Vectorized Loop (VEC_FACTOR = 4) ---
    for (uint32_t i = 0; i < M; ++i) {
        // Load 16 bytes (4 blocks of 4 bytes)
        uint8x16x4_t BQ = vld4q_u8(byte_query);

        const uint8x16_t L0_par = vdupq_n_u8(0);
        const uint8x16_t L1_par = vshlq_n_u8(BQ.val[3], 1);
        const uint8x16_t L2_par = vshlq_n_u8(BQ.val[2], 1);
        const uint8x16_t L3_par = vaddq_u8(L2_par, L1_par);
        const uint8x16_t L4_par = vshlq_n_u8(BQ.val[1], 1);
        const uint8x16_t L5_par = vaddq_u8(L4_par, L1_par);
        const uint8x16_t L6_par = vaddq_u8(L4_par, L2_par);
        const uint8x16_t L7_par = vaddq_u8(L6_par, L1_par);
        const uint8x16_t L8_par = vshlq_n_u8(BQ.val[0], 1);
        const uint8x16_t L9_par = vaddq_u8(L8_par, L1_par);
        const uint8x16_t L10_par = vaddq_u8(L8_par, L2_par);
        const uint8x16_t L11_par = vaddq_u8(L10_par, L1_par);
        const uint8x16_t L12_par = vaddq_u8(L8_par, L4_par);
        const uint8x16_t L13_par = vaddq_u8(L12_par, L1_par);
        const uint8x16_t L14_par = vaddq_u8(L12_par, L2_par);
        const uint8x16_t L15_par = vaddq_u8(L14_par, L1_par);

        uint8x16x4_t RS0, RS1, RS2, RS3;

        const uint8x16_t L0L4b = vzip1q_u8(L0_par, L4_par);
        const uint8x16_t L0L4t = vzip2q_u8(L0_par, L4_par);
        const uint8x16_t L8L12b = vzip1q_u8(L8_par, L12_par);
        const uint8x16_t L8L12t = vzip2q_u8(L8_par, L12_par);

        RS0.val[0] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L0L4b), vreinterpretq_u16_u8(L8L12b)));
        RS1.val[0] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L0L4b), vreinterpretq_u16_u8(L8L12b)));
        RS2.val[0] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L0L4t), vreinterpretq_u16_u8(L8L12t)));
        RS3.val[0] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L0L4t), vreinterpretq_u16_u8(L8L12t)));

        const uint8x16_t L1L5b = vzip1q_u8(L1_par, L5_par);
        const uint8x16_t L1L5t = vzip2q_u8(L1_par, L5_par);
        const uint8x16_t L9L13b = vzip1q_u8(L9_par, L13_par);
        const uint8x16_t L9L13t = vzip2q_u8(L9_par, L13_par);

        RS0.val[1] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L1L5b), vreinterpretq_u16_u8(L9L13b)));
        RS1.val[1] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L1L5b), vreinterpretq_u16_u8(L9L13b)));
        RS2.val[1] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L1L5t), vreinterpretq_u16_u8(L9L13t)));
        RS3.val[1] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L1L5t), vreinterpretq_u16_u8(L9L13t)));

        const uint8x16_t L2L6b = vzip1q_u8(L2_par, L6_par);
        const uint8x16_t L2L6t = vzip2q_u8(L2_par, L6_par);
        const uint8x16_t L10L14b = vzip1q_u8(L10_par, L14_par);
        const uint8x16_t L10L14t = vzip2q_u8(L10_par, L14_par);

        RS0.val[2] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L2L6b), vreinterpretq_u16_u8(L10L14b)));
        RS1.val[2] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L2L6b), vreinterpretq_u16_u8(L10L14b)));
        RS2.val[2] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L2L6t), vreinterpretq_u16_u8(L10L14t)));
        RS3.val[2] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L2L6t), vreinterpretq_u16_u8(L10L14t)));

        const uint8x16_t L3L7b = vzip1q_u8(L3_par, L7_par);
        const uint8x16_t L3L7t = vzip2q_u8(L3_par, L7_par);
        const uint8x16_t L11L15b = vzip1q_u8(L11_par, L15_par);
        const uint8x16_t L11L15t = vzip2q_u8(L11_par, L15_par);

        RS0.val[3] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L3L7b), vreinterpretq_u16_u8(L11L15b)));
        RS1.val[3] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L3L7b), vreinterpretq_u16_u8(L11L15b)));
        RS2.val[3] = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(L3L7t), vreinterpretq_u16_u8(L11L15t)));
        RS3.val[3] = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(L3L7t), vreinterpretq_u16_u8(L11L15t)));

        vst4q_u8(LUT, RS0);
        vst4q_u8(LUT + 64, RS1);
        vst4q_u8(LUT + 128, RS2);
        vst4q_u8(LUT + 192, RS3);

        // auto L0008 = vzip1q_u8(L0_par,L8_par);
        // auto L0008b = vzip2q_u8(L0_par,L8_par);
        // auto L0412 = vzip1q_u8(L4_par,L12_par);
        // auto L0412b = vzip2q_u8(L4_par,L12_par);

        // auto L0210 = vzip1q_u8(L2_par,L10_par);
        // auto L0210b = vzip2q_u8(L2_par,L10_par);
        // auto L0614 = vzip1q_u8(L6_par,L14_par);
        // auto L0614b = vzip2q_u8(L6_par,L14_par);

        // auto L0109 = vzip1q_u8(L1_par,L9_par);
        // auto L0109b = vzip2q_u8(L1_par,L9_par);
        // auto L0513 = vzip1q_u8(L5_par,L13_par);
        // auto L0513b = vzip2q_u8(L5_par,L13_par);

        // auto L0311 = vzip1q_u8(L3_par,L11_par);
        // auto L0311b = vzip2q_u8(L3_par,L11_par);
        // auto L0715 = vzip1q_u8(L7_par,L15_par);
        // auto L0715b = vzip2q_u8(L7_par,L15_par);


        // auto L0004 = vzip1q_u8(L0008,L0412);
        // auto L0004b = vzip2q_u8(L0008,L0412);
        // auto L00b04 = vzip1q_u8(L0008b,L0412b);
        // auto L00b04b = vzip2q_u8(L0008b,L0412b);

        // auto L0206 = vzip1q_u8(L0210,L0614);
        // auto L0206b = vzip2q_u8(L0210,L0614);
        // auto L02b06 = vzip1q_u8(L0210b,L0614b);
        // auto L02b06b = vzip2q_u8(L0210b,L0614b);

        // auto L0105 = vzip1q_u8(L0109,L0513);
        // auto L0105b = vzip2q_u8(L0109,L0513);
        // auto L01b05 = vzip1q_u8(L0109b,L0513b);
        // auto L01b05b = vzip2q_u8(L0109b,L0513b);

        // auto L0307 = vzip1q_u8(L0311,L0715);
        // auto L0307b = vzip2q_u8(L0311,L0715);
        // auto L03b07 = vzip1q_u8(L0311b,L0715b);
        // auto L03b07b = vzip2q_u8(L0311b,L0715b);

        // auto L0002 = vzip1q_u8(L0004,L0206);
        // auto L0002b = vzip2q_u8(L0004,L0206);
        // auto L0002c = vzip1q_u8(L0004b,L0206b);
        // auto L0002d = vzip2q_u8(L0004b,L0206b);
        // auto L0002e = vzip1q_u8(L00b04,L02b06);
        // auto L0002f = vzip2q_u8(L00b04,L02b06);
        // auto L0002g = vzip1q_u8(L00b04b,L02b06b);
        // auto L0002h = vzip2q_u8(L00b04b,L02b06b);

        // auto L0103 = vzip1q_u8(L0105,L0307);
        // auto L0103b = vzip2q_u8(L0105,L0307);
        // auto L0103c = vzip1q_u8(L0105b,L0307b);
        // auto L0103d = vzip2q_u8(L0105b,L0307b);
        // auto L0103e = vzip1q_u8(L01b05,L03b07);
        // auto L0103f = vzip2q_u8(L01b05,L03b07);
        // auto L0103g = vzip1q_u8(L01b05b,L03b07b);
        // auto L0103h = vzip2q_u8(L01b05b,L03b07b);


        // auto L0001 = vzip1q_u8(L0002,L0103);
        // auto L0001b = vzip2q_u8(L0002,L0103);
        // auto L0001c = vzip1q_u8(L0002b,L0103b);
        // auto L0001d = vzip2q_u8(L0002b,L0103b);
        // auto L0001e = vzip1q_u8(L0002c,L0103c);
        // auto L0001f = vzip2q_u8(L0002c,L0103c);
        // auto L0001g = vzip1q_u8(L0002d,L0103d);
        // auto L0001h = vzip2q_u8(L0002d,L0103d);
        // auto L0001i = vzip1q_u8(L0002e,L0103e);
        // auto L0001j = vzip2q_u8(L0002e,L0103e);
        // auto L0001k = vzip1q_u8(L0002f,L0103f);
        // auto L0001l = vzip2q_u8(L0002f,L0103f);
        // auto L0001m = vzip1q_u8(L0002g,L0103g);
        // auto L0001n = vzip2q_u8(L0002g,L0103g);
        // auto L0001o = vzip1q_u8(L0002h,L0103h);
        // auto L0001p = vzip2q_u8(L0002h,L0103h);

        // vst1q_u8(LUT,L0001);
        // vst1q_u8(LUT+16,L0001b);
        // vst1q_u8(LUT+32,L0001c);
        // vst1q_u8(LUT+48,L0001d);
        // vst1q_u8(LUT+64,L0001e);
        // vst1q_u8(LUT+80,L0001f);
        // vst1q_u8(LUT+96,L0001g);
        // vst1q_u8(LUT+112,L0001h);
        // vst1q_u8(LUT+128,L0001i);
        // vst1q_u8(LUT+144,L0001j);
        // vst1q_u8(LUT+160,L0001k);
        // vst1q_u8(LUT+176,L0001l);
        // vst1q_u8(LUT+192,L0001m);
        // vst1q_u8(LUT+208,L0001n);
        // vst1q_u8(LUT+224,L0001o);
        // vst1q_u8(LUT+240,L0001p);

        LUT        += 256; // 64 bytes
        byte_query += 64; // 16 bytes
    }
}
#else
template <uint32_t B>
inline void pack_LUT(uint8_t* byte_query, uint8_t* LUT){
    constexpr uint32_t M = B / 4;
    constexpr uint32_t pos[16]={
        3 /*0000*/, 3/*0001*/, 2/*0010*/, 3/*0011*/,
        1 /*0100*/, 3/*0101*/, 2/*0110*/, 3/*0111*/,
        0 /*1000*/, 3/*1001*/, 2/*1010*/, 3/*1011*/,
        1 /*1100*/, 3/*1101*/, 2/*1110*/, 3/*1111*/,
    };
    for(int i=0;i<M;i++){
        LUT[0] = 0;
        for(int j=1;j<16;j++){
            LUT[j] = LUT[j - lowbit(j)] + byte_query[pos[j]];
        }
        LUT        += 16;
        byte_query += 4;
    }
}
#endif

template <typename T, class TA>
inline void get_matrix_column(T* src, size_t m, size_t n, int64_t i, int64_t j, TA& dest) {
    for (int64_t k = 0; k < dest.size(); k++) {
        if (k + i >= 0 && k + i < m) {
            dest[k] = src[(k + i) * n + j];
        } 
        else {
            dest[k] = 0;
        }
    }
}

// ==============================================================
// pack 32 quantization codes in a batch from the quantization 
// codes represented by a sequence of uint8_t variables
// ==============================================================
template <uint32_t B>
void pack_codes(const uint8_t* codes, uint32_t ncode, uint8_t* blocks){
    
    uint32_t ncode_pad = (ncode + 31) / 32 * 32;
    constexpr uint32_t M = B / 4;
    const uint8_t bbs = 32;    
    memset(blocks, 0, ncode_pad * M / 2);

    const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    uint8_t* codes2 = blocks;
    for(int blk=0;blk<ncode_pad;blk+=bbs){
        for(int m=0;m<M;m+=2){
            std::array<uint8_t, 32> c, c0, c1;
            get_matrix_column(codes, ncode, M / 2, blk, m / 2, c);
            for (int j = 0; j < 32; j++) {
                c0[j] = c[j] & 15;
                c1[j] = c[j] >> 4;
            }
            for (int j = 0; j < 16; j++) {
                uint8_t d0, d1;
                d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                codes2[j] = d0;
                codes2[j + 16] = d1;
            }
            codes2 += 32;
        }
    }
}
#ifdef __aarch64__
static void krl_get_matrix_column(
        const uint8_t* src,
        size_t m,
        size_t n,
        int64_t i,
        int64_t j,
        uint8_t* dest,
        int length) {
    for (int64_t k = 0; k < length; k++) {
        if (k + i >= 0 && k + i < m) {
            dest[k] = src[(k + i) * n + j];
        } else {
            dest[k] = 0;
        }
    }
}

template<uint32_t B, uint32_t batchsize>
static void krl_pqfs_pack_codes(
        const uint8_t* codes,
        size_t ntotal,
        uint8_t* blocks) {
    if constexpr(batchsize % 32) {
        printf("Error, krl_pack_codes_4b batchsize must be a multiple of 16!\n");
        return;
    }
    if constexpr(batchsize == 0) {
        return;
    }
    constexpr int nsq = B / 4;
    constexpr int half_nsq = (nsq + 1) >> 1;
    const size_t block1 = ((ntotal + batchsize - 1) / batchsize);
    memset(blocks, 0, block1 * batchsize * half_nsq);
    for (size_t b = 0; b < block1; b++) {
        uint8_t* codes2 = blocks + b * batchsize * half_nsq; 
        const int64_t i_base = b * batchsize;
        uint8_t c[batchsize], c0[batchsize], c1[batchsize];
        for (int sq = 0; sq < half_nsq; sq++) { 
            krl_get_matrix_column(codes, ntotal, half_nsq, i_base, sq, c, batchsize);
            for (int j = 0; j < batchsize; j++) {
                    c0[j] = c[j] & 15;  /* base vector dim0 */
                    c1[j] = c[j] >> 4;  /* base vector dim1 */
            }
            for (int j = 0; j < batchsize; j += 16) {
                for(int k = 0; k < 8; ++k) {  /* vector id */
                    uint8_t d0 = c0[k + j] | (c0[k + j + 8] << 4);  /* dim0: 0,8 1,9 2,10 ... */
                    uint8_t d1 = c1[k + j] | (c1[k + j + 8] << 4);  /* dim1: 0,8 1,9 2,10 ... */
                    codes2[j + (k << 1)] |= d0;           
                    codes2[j + (k << 1) + 1] |= d1;  /* dim0，dim1，dim0，dim1，... */
                }
            }
            codes2 += batchsize;
        }
    }
}
// ==============================================================
// pack 32 quantization codes in a batch from the quantization 
// codes represented by a sequence of uint64_t variables
// ==============================================================
template<uint32_t B, uint32_t batchsize = 32>
void pack_codes(const uint64_t* binary_code, uint32_t ncode, uint8_t* blocks){
    uint32_t ncode_pad = (ncode + batchsize - 1) / batchsize * batchsize;
    memset(blocks, 0, ncode_pad * sizeof(uint8_t));

    uint8_t * binary_code_8bit = new uint8_t [ncode_pad * B / 8];
    memcpy(binary_code_8bit, binary_code, ncode * B / 64 * sizeof(uint64_t));

    for(int i=0;i<ncode;i++)
        for(int j=0;j<B/64;j++)
            for(int k=0;k<4;k++)
                swap(binary_code_8bit[i * B / 8 + 8 * j + k], binary_code_8bit[i * B / 8 + 8 * j + 8 - k - 1]);

    for(int i=0;i<ncode*B/8;i++){
        uint8_t v = binary_code_8bit[i];
        uint8_t x = (v >> 4);
        uint8_t y = (v & 15);
        binary_code_8bit[i] = (y << 4 | x);
    }
    krl_pqfs_pack_codes<B, batchsize>(binary_code_8bit, ncode, blocks);
    delete[] binary_code_8bit;
}
#else
template<uint32_t B>
void pack_codes(const uint64_t* binary_code, uint32_t ncode, uint8_t* blocks){
    uint32_t ncode_pad = (ncode + 31) / 32 * 32;
    memset(blocks, 0, ncode_pad * sizeof(uint8_t));

    uint8_t * binary_code_8bit = new uint8_t [ncode_pad * B / 8];
    memcpy(binary_code_8bit, binary_code, ncode * B / 64 * sizeof(uint64_t));

    for(int i=0;i<ncode;i++)
        for(int j=0;j<B/64;j++)
            for(int k=0;k<4;k++)
                swap(binary_code_8bit[i * B / 8 + 8 * j + k], binary_code_8bit[i * B / 8 + 8 * j + 8 - k - 1]);

    for(int i=0;i<ncode*B/8;i++){
        uint8_t v = binary_code_8bit[i];
        uint8_t x = (v >> 4);
        uint8_t y = (v & 15);
        binary_code_8bit[i] = (y << 4 | x);
    }
    pack_codes<B>(binary_code_8bit, ncode, blocks);
    delete [] binary_code_8bit;
}
#endif


