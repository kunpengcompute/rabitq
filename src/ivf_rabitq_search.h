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
// ==================================================================
// IVFRN() involves pre-processing steps (e.g., packing the
// quantization codes into a batch) in the index phase.

// search() is the main function of the query phase.
// ==================================================================
#pragma once
#ifdef __aarch64__
#include "ivf_rabitq.h"

#if DIM >= 128
    constexpr uint32_t SIZE = 64;
#else 
    constexpr uint32_t SIZE = 96;
#endif

constexpr uint32x4_t simd_offset = {0x11111111, 0x22222222, 0x44444444, 0x88888888};
constexpr uint32x4_t simd_offset_16 = {0x1111, 0x2222, 0x4444, 0x8888};

// scan impl
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint64_t *quant_query, uint64_t *ptr_binary_code,  uint32_t len, Factor *ptr_fac, \
                        const float sqr_y, const float vl, const float width, const float sumq, \
                        float *query, float *data, uint32_t *id){

    float y = std::sqrt(sqr_y);
    float res[SIZE];
    float *ptr_res = &res[0];
    int it = len / SIZE;

    for(int i=0;i<it;i++){   
        ptr_res = &res[0];
        for (int j = 0; j < SIZE; j++){
            float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + (space.ip_byte_bin(quant_query, ptr_binary_code) * 2 - sumq) * (ptr_fac -> factor_ip) * width;
            float error_bound = y * (ptr_fac -> error);
            *ptr_res = tmp_dist - error_bound;
            ptr_binary_code += B / 64;
            ptr_fac ++;
            ptr_res ++;
        }

        ptr_res = &res[0];
        for(int j=0;j<SIZE;j++){
            if(*ptr_res < distK){

                float gt_dist = sqr_dist<D>(query, data);
                if(gt_dist < distK){
                    KNNs.emplace(gt_dist, *id);
                    if(KNNs.size() > k) KNNs.pop();
                    if(KNNs.size() == k)distK = KNNs.top().first;
        }
            }
            data += D;
            ptr_res++;
            id++;
        }
    }

    ptr_res = &res[0];
    for(int i=it * SIZE;i<len;i++){
        float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + (space.ip_byte_bin(quant_query, ptr_binary_code) * 2 -sumq) * (ptr_fac -> factor_ip) * width;
        float error_bound = y * (ptr_fac -> error);
        *ptr_res = tmp_dist - error_bound;
        ptr_binary_code += B / 64;
        ptr_fac ++;
        ptr_res ++;
    }

    ptr_res = &res[0];
    for(int i=it * SIZE;i<len;i++){
        if(*ptr_res < distK){
            float gt_dist = sqr_dist<D>(query, data);
            if(gt_dist < distK){
                KNNs.emplace(gt_dist, *id);
                if(KNNs.size() > k) KNNs.pop();
                if(KNNs.size() == k)distK = KNNs.top().first;
            }
        }
        data += D;
        ptr_res++;
        id++;
    }
}

#define PROCESS_32_LOW_DIST(ID)                                             \
{                                                                           \
        const float32x4_t v_ld = vdupq_n_f32(local_distK);                  \
        uint16x8_t tbl_u16_0 = vld1q_u16(result + ID);                      \
        uint16x8_t tbl_u16_1 = vld1q_u16(result + ID + 8);                  \
        float16x8_t v_sqrx0 = vld1q_f16(ptr_fac->sqr_x);                    \
        float16x8_t v_sqrx1 = vld1q_f16(ptr_fac->sqr_x + 8);                \
        float32x4_t result0 = vcvt_f32_f16(vget_low_f16(v_sqrx0));          \
        float32x4_t result1 = vcvt_high_f32_f16(v_sqrx0);                   \
        float32x4_t result2 = vcvt_f32_f16(vget_low_f16(v_sqrx1));          \
        float32x4_t result3 = vcvt_high_f32_f16(v_sqrx1);                   \
        float16x8_t tbl_f16_0 = vcvtq_f16_u16(tbl_u16_0);                   \
        float16x8_t tbl_f16_1 = vcvtq_f16_u16(tbl_u16_1);                   \
        float16x8_t v_ppc0 = vld1q_f16(ptr_fac->factor_ppc);                \
        float16x8_t v_ppc1 = vld1q_f16(ptr_fac->factor_ppc + 8);            \
        result0 = vfmlalq_low_f16( result0, v_ppc0, v_vl);                  \
        result1 = vfmlalq_high_f16(result1, v_ppc0, v_vl);                  \
        result2 = vfmlalq_low_f16( result2, v_ppc1, v_vl);                  \
        result3 = vfmlalq_high_f16(result3, v_ppc1, v_vl);                  \
        float16x8_t v_ip0 = vld1q_f16(ptr_fac->factor_ip);                  \
        float16x8_t v_ip1 = vld1q_f16(ptr_fac->factor_ip + 8);              \
        tbl_f16_0 = vsubq_f16(tbl_f16_0, v_sumq);                           \
        tbl_f16_1 = vsubq_f16(tbl_f16_1, v_sumq);                           \
        float16x8_t v_error0 = vld1q_f16(ptr_fac->error);                   \
        float16x8_t v_error1 = vld1q_f16(ptr_fac->error + 8);               \
        ptr_fac++;                                                          \
        float32x4_t tmp_lo0 = vfmlalq_low_f16( vdupq_n_f32(0.0f), v_ip0, tbl_f16_0);  \
        float32x4_t tmp_hi0 = vfmlalq_high_f16(vdupq_n_f32(0.0f), v_ip0, tbl_f16_0);  \
        float32x4_t tmp_lo1 = vfmlalq_low_f16( vdupq_n_f32(0.0f), v_ip1, tbl_f16_1);  \
        float32x4_t tmp_hi1 = vfmlalq_high_f16(vdupq_n_f32(0.0f), v_ip1, tbl_f16_1);  \
        result0 = vfmlslq_low_f16( result0, v_error0, v_y);                 \
        result1 = vfmlslq_high_f16(result1, v_error0, v_y);                 \
        result2 = vfmlslq_low_f16( result2, v_error1, v_y);                 \
        result3 = vfmlslq_high_f16(result3, v_error1, v_y);                 \
        tbl_u16_0 = vld1q_u16(result + ID + 16);                            \
        tbl_u16_1 = vld1q_u16(result + ID + 24);                            \
        result0 = vfmaq_f32(result0, tmp_lo0, v_width);                     \
        result1 = vfmaq_f32(result1, tmp_hi0, v_width);                     \
        result2 = vfmaq_f32(result2, tmp_lo1, v_width);                     \
        result3 = vfmaq_f32(result3, tmp_hi1, v_width);                     \
        v_sqrx0 = vld1q_f16(ptr_fac->sqr_x);                                \
        v_sqrx1 = vld1q_f16(ptr_fac->sqr_x + 8);                            \
        uint32x4_t mask0 = vcltq_f32(result0, v_ld);                        \
        uint32x4_t mask1 = vcltq_f32(result1, v_ld);                        \
        uint32x4_t mask2 = vcltq_f32(result2, v_ld);                        \
        uint32x4_t mask3 = vcltq_f32(result3, v_ld);                        \
                                                                            \
        tbl_f16_0 = vcvtq_f16_u16(tbl_u16_0);                               \
        tbl_f16_1 = vcvtq_f16_u16(tbl_u16_1);                               \
        result0 = vcvt_f32_f16(vget_low_f16(v_sqrx0));                      \
        result2 = vcvt_f32_f16(vget_low_f16(v_sqrx1));                      \
        result1 = vcvt_high_f32_f16(v_sqrx0);                               \
        result3 = vcvt_high_f32_f16(v_sqrx1);                               \
        mask0 = vsliq_n_u32(mask0, mask1, 4);                               \
        mask2 = vsliq_n_u32(mask2, mask3, 4);                               \
        v_ppc0 = vld1q_f16(ptr_fac->factor_ppc);                            \
        v_ppc1 = vld1q_f16(ptr_fac->factor_ppc + 8);                        \
        result0 = vfmlalq_low_f16( result0, v_ppc0, v_vl);                  \
        result1 = vfmlalq_high_f16(result1, v_ppc0, v_vl);                  \
        result2 = vfmlalq_low_f16( result2, v_ppc1, v_vl);                  \
        result3 = vfmlalq_high_f16(result3, v_ppc1, v_vl);                  \
        v_ip0 = vld1q_f16(ptr_fac->factor_ip);                              \
        v_ip1 = vld1q_f16(ptr_fac->factor_ip + 8);                          \
        tbl_f16_0 = vsubq_f16(tbl_f16_0, v_sumq);                           \
        tbl_f16_1 = vsubq_f16(tbl_f16_1, v_sumq);                           \
        v_error0 = vld1q_f16(ptr_fac->error);                               \
        v_error1 = vld1q_f16(ptr_fac->error + 8);                           \
        ptr_fac++;                                                          \
        mask0 = vsliq_n_u32(mask0, mask2, 8);                               \
        tmp_lo0 = vfmlalq_low_f16( vdupq_n_f32(0.0f), v_ip0, tbl_f16_0);    \
        tmp_hi0 = vfmlalq_high_f16(vdupq_n_f32(0.0f), v_ip0, tbl_f16_0);    \
        tmp_lo1 = vfmlalq_low_f16( vdupq_n_f32(0.0f), v_ip1, tbl_f16_1);    \
        tmp_hi1 = vfmlalq_high_f16(vdupq_n_f32(0.0f), v_ip1, tbl_f16_1);    \
        result0 = vfmlslq_low_f16( result0, v_error0, v_y);                 \
        result1 = vfmlslq_high_f16(result1, v_error0, v_y);                 \
        result2 = vfmlslq_low_f16( result2, v_error1, v_y);                 \
        result3 = vfmlslq_high_f16(result3, v_error1, v_y);                 \
        result0 = vfmaq_f32(result0, tmp_lo0, v_width);                     \
        result1 = vfmaq_f32(result1, tmp_hi0, v_width);                     \
        result2 = vfmaq_f32(result2, tmp_lo1, v_width);                     \
        result3 = vfmaq_f32(result3, tmp_hi1, v_width);                     \
        uint32x4_t mask4 = vcltq_f32(result0, v_ld);                        \
                   mask1 = vcltq_f32(result1, v_ld);                        \
                   mask2 = vcltq_f32(result2, v_ld);                        \
                   mask3 = vcltq_f32(result3, v_ld);                        \
        mask4 = vsliq_n_u32(mask4, mask1, 4);                               \
        mask2 = vsliq_n_u32(mask2, mask3, 4);                               \
        mask4 = vsliq_n_u32(mask4, mask2, 8);                               \
        mask0 = vsliq_n_u32(mask0, mask4, 16);                              \
        mask0 = vandq_u32(mask0, simd_offset);                              \
        cnt = vaddvq_u32(mask0);                                            \
}

#define PROCESS_16_LOW_DIST(ID)                                             \
{                                                                           \
        const float32x4_t v_ld = vdupq_n_f32(local_distK);                  \
        uint16x8_t tbl_u16_0 = vld1q_u16(result + ID);                      \
        uint16x8_t tbl_u16_1 = vld1q_u16(result + ID + 8);                  \
        float16x8_t v_sqrx0 = vld1q_f16(ptr_fac->sqr_x);                    \
        float16x8_t v_sqrx1 = vld1q_f16(ptr_fac->sqr_x + 8);                \
        float32x4_t result0 = vcvt_f32_f16(vget_low_f16(v_sqrx0));          \
        float32x4_t result1 = vcvt_high_f32_f16(v_sqrx0);                   \
        float32x4_t result2 = vcvt_f32_f16(vget_low_f16(v_sqrx1));          \
        float32x4_t result3 = vcvt_high_f32_f16(v_sqrx1);                   \
        float16x8_t tbl_f16_0 = vcvtq_f16_u16(tbl_u16_0);                   \
        float16x8_t tbl_f16_1 = vcvtq_f16_u16(tbl_u16_1);                   \
        float16x8_t v_ppc0 = vld1q_f16(ptr_fac->factor_ppc);                \
        float16x8_t v_ppc1 = vld1q_f16(ptr_fac->factor_ppc + 8);            \
        result0 = vfmlalq_low_f16( result0, v_ppc0, v_vl);                  \
        result1 = vfmlalq_high_f16(result1, v_ppc0, v_vl);                  \
        result2 = vfmlalq_low_f16( result2, v_ppc1, v_vl);                  \
        result3 = vfmlalq_high_f16(result3, v_ppc1, v_vl);                  \
        float16x8_t v_ip0 = vld1q_f16(ptr_fac->factor_ip);                  \
        float16x8_t v_ip1 = vld1q_f16(ptr_fac->factor_ip + 8);              \
        tbl_f16_0 = vsubq_f16(tbl_f16_0, v_sumq);                           \
        tbl_f16_1 = vsubq_f16(tbl_f16_1, v_sumq);                           \
        float16x8_t v_error0 = vld1q_f16(ptr_fac->error);                   \
        float16x8_t v_error1 = vld1q_f16(ptr_fac->error + 8);               \
        float32x4_t tmp_lo0 = vfmlalq_low_f16( vdupq_n_f32(0.0f), v_ip0, tbl_f16_0);  \
        float32x4_t tmp_hi0 = vfmlalq_high_f16(vdupq_n_f32(0.0f), v_ip0, tbl_f16_0);  \
        float32x4_t tmp_lo1 = vfmlalq_low_f16( vdupq_n_f32(0.0f), v_ip1, tbl_f16_1);  \
        float32x4_t tmp_hi1 = vfmlalq_high_f16(vdupq_n_f32(0.0f), v_ip1, tbl_f16_1);  \
        result0 = vfmlslq_low_f16( result0, v_error0, v_y);                 \
        result1 = vfmlslq_high_f16(result1, v_error0, v_y);                 \
        result2 = vfmlslq_low_f16( result2, v_error1, v_y);                 \
        result3 = vfmlslq_high_f16(result3, v_error1, v_y);                 \
        result0 = vfmaq_f32(result0, tmp_lo0, v_width);                     \
        result1 = vfmaq_f32(result1, tmp_hi0, v_width);                     \
        result2 = vfmaq_f32(result2, tmp_lo1, v_width);                     \
        result3 = vfmaq_f32(result3, tmp_hi1, v_width);                     \
        ptr_fac++;                                                          \
        uint32x4_t mask0 = vcltq_f32(result0, v_ld);                        \
        uint32x4_t mask1 = vcltq_f32(result1, v_ld);                        \
        uint32x4_t mask2 = vcltq_f32(result2, v_ld);                        \
        uint32x4_t mask3 = vcltq_f32(result3, v_ld);                        \
        mask0 = vsliq_n_u32(mask0, mask1, 4);                               \
        mask2 = vsliq_n_u32(mask2, mask3, 4);                               \
        mask0 = vsliq_n_u32(mask0, mask2, 8);                               \
        mask0 = vandq_u32(mask0, simd_offset_16);                           \
        cnt = vaddvq_u32(mask0);                                            \
}

#define COMPARE_MASK_SOAR(j, cnt) \
    if (cnt) {                                                                      \
        {                                                                           \
            int cnt_count = __builtin_ctz(cnt);                                     \
            cnt &= (cnt - 1);                                                       \
            int32_t vid = id[j + cnt_count];                                        \
            if (seen.count(vid));                                                   \
            else {                                                                  \
                float gt = krl_L2sqr_f16f32<D>(query, data + (j + cnt_count) * D);  \
                if (gt < distK) {                                                   \
                    seen.emplace(vid);                                              \
                    KNNs.emplace(gt, vid);                                          \
                    if (KNNs.size() > k) {                                          \
                        KNNs.pop();                                                 \
    }                                                               \
                    distK = KNNs.top().first;                                       \
                }                                                                   \
            }                                                                       \
        }                                                                           \
        while (cnt) {                                                               \
            int cnt_count = __builtin_ctz(cnt);                                     \
            cnt &= (cnt - 1);                                                       \
            int32_t vid = id[j + cnt_count];                                        \
            if (seen.count(vid)) continue;                                          \
            float gt = krl_L2sqr_f16f32<D>(query, data + (j + cnt_count) * D);      \
            if (gt < distK) {                                                       \
                seen.emplace(vid);                                                  \
                KNNs.emplace(gt, vid);                                              \
                if (KNNs.size() > k) {                                              \
                    KNNs.pop();                                                     \
                }                                                                   \
                distK = KNNs.top().first;                                           \
            }                                                                       \
        }                                                                           \
        local_distK = (distK - sqr_y) * low_dist_scale;                             \
    }

#define PD4 2
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::fast_scan_mask_soar(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code,  uint32_t len, Factor_f16 *ptr_fac, \
                        const float sqr_y, const float16_t vl, const float width, const float16_t sumq, \
                        float16_t *query, float16_t *data, uint32_t *id, std::unordered_set<uint32_t> &seen, float low_dist_scale) {
    float local_distK = (distK - sqr_y) * low_dist_scale; 
    float16_t y = std::sqrt(sqr_y);
    uint32_t it = len / SIZE;
    uint32_t remain = len % SIZE;
    uint16_t PORTABLE_ALIGN64 result[SIZE];

    const float16x8_t v_vl = vdupq_n_f16(vl);
    const float32x4_t v_width = vdupq_n_f32(width);
    const float16x8_t v_sumq = vdupq_n_f16(sumq);
    const float16x8_t v_y = vdupq_n_f16(y);

    while(it--) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        packed_code += SIZE * B / 8;

        for (int j = 0; j < SIZE; j += 32) { 
            uint32_t cnt;
            __builtin_prefetch(ptr_fac + PD4 * 2, 0, 3);
            __builtin_prefetch(ptr_fac + PD4 * 2 + 1, 0, 3);
            PROCESS_32_LOW_DIST(j);

            COMPARE_MASK_SOAR(j, cnt);
        }

        data += SIZE * D;
        id   += SIZE;
    }
    if (remain) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        int last_remain = remain & 15;
        remain &= (-16);

        for (int i = 0; i < remain; i += 16) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(i); 

            COMPARE_MASK_SOAR(i, cnt);
        }
        data += remain * D;
        id += remain;
        if (last_remain) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(remain); 
            cnt &= ((1 << last_remain) - 1);

            COMPARE_MASK_SOAR(0, cnt);
        }
    }
}

#define COMPARE_SOAR(j, cnt) \
    if (cnt) {                                                                          \
        {                                                                               \
            int cnt_count = __builtin_ctz(cnt);                                         \
            cnt &= (cnt - 1);                                                           \
            int32_t vid = id[cnt_count + j];                                            \
            if (seen.count(vid));                                                       \
            else {                                                                      \
                float gt_dist = krl_L2sqr_f16f32<D>(query, data + (cnt_count + j) * D); \
                if (gt_dist < distK) {                                                  \
                    seen.emplace(vid);                                                  \
                    KNNs.emplace(gt_dist, vid);                                         \
                    if (KNNs.size() > k) {                                              \
                        KNNs.pop();                                                     \
                        distK = KNNs.top().first;                                       \
                    }                                                                   \
                }                                                                       \
            }                                                                           \
        }                                                                               \
        while (cnt) {                                                                   \
            int cnt_count = __builtin_ctz(cnt);                                         \
            cnt &= (cnt - 1);                                                           \
            int32_t vid = id[cnt_count + j];                                            \
            if (seen.count(vid)) continue;                                              \
            float gt_dist = krl_L2sqr_f16f32<D>(query, data + (cnt_count + j) * D);     \
            if (gt_dist < distK) {                                                      \
                seen.emplace(vid);                                                      \
                KNNs.emplace(gt_dist, vid);                                             \
                if (KNNs.size() > k) {                                                  \
                    KNNs.pop();                                                         \
                    distK = KNNs.top().first;                                           \
                }                                                                       \
            }                                                                           \
        }                                                                               \
        local_distK = (distK - sqr_y) * low_dist_scale;                                 \
    }

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::fast_scan_soar(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code,  uint32_t len, Factor_f16 *ptr_fac, \
                        const float sqr_y, const float16_t vl, const float width, const float16_t sumq, \
                        float16_t *query, float16_t *data, uint32_t *id, std::unordered_set<uint32_t> &seen, float low_dist_scale){

    float local_distK = (distK - sqr_y) * low_dist_scale;

    float16_t y = std::sqrt(sqr_y);
    uint32_t it = len / SIZE;
    uint32_t remain = len % SIZE;
    uint16_t PORTABLE_ALIGN64 result[SIZE];

    const float16x8_t v_vl = vdupq_n_f16(vl);
    const float32x4_t v_width = vdupq_n_f32(width);
    const float16x8_t v_sumq = vdupq_n_f16(sumq);
    const float16x8_t v_y = vdupq_n_f16(y);

    while(it--) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        packed_code += SIZE * B / 8;

        for (int j = 0; j < SIZE; j += 32) {
            uint32_t cnt;
            __builtin_prefetch(ptr_fac + PD4 * 2, 0, 3);
            __builtin_prefetch(ptr_fac + PD4 * 2 + 1, 0, 3);
            PROCESS_32_LOW_DIST(j); 

            COMPARE_SOAR(j, cnt);
        }
        data += SIZE * D;
        id   += SIZE;
    }
    if (remain) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        int last_remain = remain & 15;
        remain &= (-16);

        for (int i = 0; i < remain; i += 16) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(i);

            COMPARE_SOAR(i, cnt);
        }
        data += remain * D;
        id += remain;
        if (last_remain) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(remain); 
            cnt &= ((1 << last_remain) - 1);

            COMPARE_SOAR(0, cnt);
        }
    }
}

#define COMPARE_MASK(j, cnt) \
    if(cnt) {                                                                   \
        {                                                                       \
            int cnt_count = __builtin_ctz(cnt);                                 \
            cnt &= (cnt - 1);                                                   \
            float gt = krl_L2sqr_f16f32<D>(query, data + (j + cnt_count) * D);  \
            if (gt < distK) {                                                   \
                KNNs.emplace(gt, id[j + cnt_count]);                            \
                if (KNNs.size() > k) {                                          \
                    KNNs.pop();                                                 \
                    distK = KNNs.top().first;                                   \
                }                                                               \
            }                                                                   \
        }                                                                       \
        while (cnt) {                                                           \
            int cnt_count = __builtin_ctz(cnt);                                 \
            cnt &= (cnt - 1);                                                   \
            float gt = krl_L2sqr_f16f32<D>(query, data + (j + cnt_count) * D);  \
            if (gt < distK) {                                                   \
                KNNs.emplace(gt, id[j + cnt_count]);                            \
                if (KNNs.size() > k) {                                          \
                    KNNs.pop();                                                 \
                    distK = KNNs.top().first;                                   \
                }                                                               \
            }                                                                   \
        }                                                                       \
        local_distK = (distK - sqr_y) * low_dist_scale;                         \
    }

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::fast_scan_mask(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code,  uint32_t len, Factor_f16 *ptr_fac, \
                        const float sqr_y, const float16_t vl, const float width, const float16_t sumq, \
                        float16_t *query, float16_t *data, uint32_t *id, float low_dist_scale) {

    float local_distK = (distK - sqr_y) * low_dist_scale; 

    float16_t y = std::sqrt(sqr_y);
    uint32_t it = len / SIZE;
    uint32_t remain = len % SIZE;
    uint16_t PORTABLE_ALIGN64 result[SIZE];

    const float16x8_t v_vl = vdupq_n_f16(vl);
    const float32x4_t v_width = vdupq_n_f32(width);
    const float16x8_t v_sumq = vdupq_n_f16(sumq);
    const float16x8_t v_y = vdupq_n_f16(y);

    while(it--) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        packed_code += SIZE * B / 8;

        for (int j = 0; j < SIZE; j += 32) {
            uint32_t cnt;
            __builtin_prefetch(ptr_fac + PD4 * 2, 0, 3);
            __builtin_prefetch(ptr_fac + PD4 * 2 + 1, 0, 3);
            PROCESS_32_LOW_DIST(j);

            COMPARE_MASK(j, cnt);
        }

        data += SIZE * D;
        id   += SIZE;
    }
    if (remain) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        int last_remain = remain & 15;
        remain &= (-16);

        for (int i = 0; i < remain; i += 16) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(i); 

            COMPARE_MASK(i, cnt);
        }
        data += remain * D;
        id += remain;

    if (last_remain) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(remain); 
            cnt &= ((1 << last_remain) - 1);

            COMPARE_MASK(0, cnt);
        }
    }
}

#define COMPARE_DEFAULT(j, cnt) \
    if (cnt) {                                                                      \
        {                                                                           \
            int cnt_count = __builtin_ctz(cnt);                                     \
            cnt &= (cnt - 1);                                                       \
            float gt_dist = krl_L2sqr_f16f32<D>(query, data + (j + cnt_count) * D); \
            if (gt_dist < distK) {                                                  \
                KNNs.emplace(gt_dist, id[j + cnt_count]);                           \
                if (KNNs.size() > k) {                                              \
                    KNNs.pop();                                                     \
                    distK = KNNs.top().first;                                       \
                }                                                                   \
            }                                                                       \
        }                                                                           \
        while (cnt) {                                                               \
            int cnt_count = __builtin_ctz(cnt);                                     \
            cnt &= (cnt - 1);                                                       \
            float gt_dist = krl_L2sqr_f16f32<D>(query, data + (j + cnt_count) * D); \
            if (gt_dist < distK) {                                                  \
                KNNs.emplace(gt_dist, id[j + cnt_count]);                           \
                if (KNNs.size() > k) {                                              \
                    KNNs.pop();                                                     \
                    distK = KNNs.top().first;                                       \
                }                                                                   \
            }                                                                       \
        }                                                                           \
        local_distK = (distK - sqr_y) * low_dist_scale;                             \
    }

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::fast_scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code,  uint32_t len, Factor_f16 *ptr_fac, \
                        const float sqr_y, const float16_t vl, const float width, const float16_t sumq, \
                        float16_t *query, float16_t *data, uint32_t *id, float low_dist_scale){

    float local_distK = (distK - sqr_y) * low_dist_scale;

    float16_t y = std::sqrt(sqr_y);
    uint32_t it = len / SIZE;
    uint32_t remain = len % SIZE;
    uint16_t PORTABLE_ALIGN64 result[SIZE];

    const float16x8_t v_vl = vdupq_n_f16(vl);
    const float32x4_t v_width = vdupq_n_f32(width);
    const float16x8_t v_sumq = vdupq_n_f16(sumq);
    const float16x8_t v_y = vdupq_n_f16(y);

    while(it--) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        packed_code += SIZE * B / 8;
        for (int j = 0; j < SIZE; j += 32) {
            uint32_t cnt;
            __builtin_prefetch(ptr_fac + PD4 * 2, 0, 3);
            __builtin_prefetch(ptr_fac + PD4 * 2 + 1, 0, 3);
            PROCESS_32_LOW_DIST(j); 

            COMPARE_DEFAULT(j, cnt);
        }
        data += SIZE * D;
        id   += SIZE;
    }
    if (remain) {
        accumulate<B, SIZE>(packed_code, LUT, result);
        int last_remain = remain & 15;
        remain &= (-16);

        for (int i = 0; i < remain; i += 16) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(i); 

            COMPARE_DEFAULT(i, cnt);
        }
        data += remain * D;
        id += remain;
        if (last_remain) {
            uint32_t cnt;
            PROCESS_16_LOW_DIST(remain); 
            cnt &= ((1 << last_remain) - 1);

            COMPARE_DEFAULT(0, cnt);
        }
    }
}

#define PD3 30
// search impl
template <uint32_t D, uint32_t B>
ResultHeap IVFRN<D, B>::search(float* query, float* rd_query, uint32_t k, uint32_t nprobe,
                            float soar_lambda, float threshold, int pred_nprobe, PredictFunc pred, float distK) const{
    if (query == nullptr || rd_query == nullptr) {
        throw std::invalid_argument("search: query pointers must not be null");
    }
    if (k == 0) {
        throw std::invalid_argument("search: k must be greater than 0");
    }
    if (C == 0 || C > numC || centroid == nullptr || centroid_f16 == nullptr ||
        u == nullptr || start == nullptr || len == nullptr || id == nullptr) {
        throw std::runtime_error("search: index is not initialized");
    }
    if (nprobe == 0 || nprobe > C) {
        throw std::invalid_argument("search: nprobe is out of range");
    }
    if (soar_lambda > 0 && !use_soar) {
        throw std::invalid_argument("search: SOAR is not enabled for this index");
    }
    if (pred_nprobe > 0) {
        if (pred == nullptr) {
            throw std::invalid_argument("search: pred callback must not be null when pred_nprobe is enabled");
        }
        if (pred_nprobe > static_cast<int>(C)) {
            throw std::invalid_argument("search: pred_nprobe is out of range");
        }
    }
#if defined(FAST_SCAN)
    if (centroid_f16 == nullptr || data_f16 == nullptr || packed_code == nullptr || fac_f16_start == nullptr) {
        throw std::runtime_error("search: FAST_SCAN buffers are not initialized");
    }
    if (use_soar && (start_spilled == nullptr || len_spilled == nullptr || id_spilled == nullptr ||
                     data_f16_spilled == nullptr || packed_code_spilled == nullptr || fac_f16_start_spilled == nullptr)) {
        throw std::runtime_error("search: SOAR buffers are not initialized");
    }
#elif defined(SCAN)
    if (binary_code == nullptr || fac == nullptr || data == nullptr) {
        throw std::runtime_error("search: SCAN buffers are not initialized");
    }
    if (use_soar && (start_spilled == nullptr || len_spilled == nullptr || id_spilled == nullptr ||
                     binary_code_spilled == nullptr || fac_spilled == nullptr || data_spilled == nullptr)) {
        throw std::runtime_error("search: SOAR buffers are not initialized");
    }
#endif
    //// model
    if (pred_nprobe > 0) {
        vector<Entry> fdata(4);
        Eigen::Map<const Eigen::RowVectorXf> query_vec(query, D);
        float mean_val = query_vec.mean();
        float std_val = std::sqrt((query_vec.array() - mean_val).square().mean());
        int nonzero_count = (query_vec.array() != 0).count();
        float sparsity = static_cast<float>(nonzero_count) / D;
        float norm = query_vec.norm();

        fdata[0].fvalue = mean_val;
        fdata[1].fvalue = std_val;
        fdata[2].fvalue = sparsity;
        fdata[3].fvalue = norm;

        double tmp = 0;
        pred(fdata.data(), 0, &tmp);
        if (tmp <= threshold){
            nprobe = pred_nprobe;
        } 
    }
    //// model

    // ===========================================================================================================
    // Find out the nearest N_{probe} centroids to the query vector.
    Result centroid_dist[numC];
    constexpr uint32_t D_B_max = D > B ? D : B;
    float16_t* query_f16 = static_cast<float16_t*>(upper_bound_aligned_alloc(64, D_B_max * sizeof(float16_t)));
    if (query_f16 == nullptr) {
        throw std::runtime_error("search: failed to allocate query buffer");
    }
    quant_f16(rd_query, B, query_f16);
    for(int i = 0; i < C; i++) {
        __builtin_prefetch(centroid_f16 + B * (i + PD3), 0, 3);
        centroid_dist[i].first = krl_L2sqr_f16f32<B>(query_f16, centroid_f16 + B * i);
        centroid_dist[i].second = i;
    }
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);
    // ===========================================================================================================
#if defined(FAST_SCAN)
    quant_f16(query, D, query_f16);
    ResultHeap ret;
    if (soar_lambda > 0){
        ret = search_fast_scan<true>(centroid_dist, query_f16, rd_query, nprobe, k, distK);
    } else {
        ret = search_fast_scan<false>(centroid_dist, query_f16, rd_query, nprobe, k, distK);
    }
    free(query_f16);
    return ret;
#elif defined(SCAN)
    ResultHeap KNNs;
    free(query_f16);
    // Scan the first nprobe clusters.
    const Result *ptr_centroid_dist = (&centroid_dist[0]);
    uint8_t PORTABLE_ALIGN64 byte_query[B];
    uint8_t PORTABLE_ALIGN64 LUT[B / 4 * 16];
    thread_local std::unordered_set<uint32_t> seen;
    seen.clear();
    for(int pb=0;pb<nprobe;pb++){
        QUANT_4BIT()
        uint64_t PORTABLE_ALIGN64 quant_query[B_QUERY * B / 64];
        memset(quant_query, 0, sizeof(quant_query));
        space.transpose_bin(byte_query, quant_query);
        scan(KNNs, distK, k,\
                quant_query, binary_code + 1ull * start[c] * (B / 64), len[c], fac + start[c], \
                sqr_y, vl, width, sum_q,\
                query, data + 1ull * start[c] * D, id + start[c]);
    }
    return KNNs;
#endif    
    std::cerr << "Error, Undefined branch!" << std::endl;
    ResultHeap KNNs;
    return KNNs;
}
#endif