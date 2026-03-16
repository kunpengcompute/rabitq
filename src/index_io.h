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
#ifdef __aarch64__
#include "ivf_rabitq.h"

#define FREE_DATA(X)    \
    if(X != nullptr)    \
        std::free(X);   \
    X = nullptr;

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::pack_codes_from_file() {
#if defined(FAST_SCAN)
    packed_start = new uint32_t [C];
    int cur = 0;
    for(int i=0;i<C;i++){
        packed_start[i] = cur;
        cur += (len[i] + SIZE - 1) / SIZE * SIZE * (B / 8);
    }

    packed_code = static_cast<uint8_t*>(upper_bound_aligned_alloc(32, cur * sizeof(uint8_t)));
    for(int i=0;i<C;i++){
        pack_codes<B, SIZE>(binary_code + 1ull * start[i] * (B / 64), len[i], packed_code + 1ull * packed_start[i]);
    }
#else 
    packed_start = NULL;
    packed_code  = NULL;
#endif
}

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::compute_factor() {
    float _max = 0;
    float _min = 3.4e38;
    
    // compute f32 factor
    Factor* cur_fac = fac;
    for(int i = 0; i < C; ++i) {
        fac_start[i] = cur_fac;
        for (int j = 0; j < len[i]; j += 4) {
            for (int k = 0; k < 4; ++k) {
                if (j + k < len[i]) {
                    int cid = start[i] + j + k;
                    long double x_x0 = (long double) dist_to_c[cid] / x0[cid];
                    (*cur_fac).sqr_x[k] = dist_to_c[cid] * dist_to_c[cid];
                    (*cur_fac).error[k] = 2 * max_x1 * std::sqrt(x_x0 * x_x0 - dist_to_c[cid] * dist_to_c[cid]);
                    (*cur_fac).factor_ppc[k] = -2 / fac_norm * x_x0  * ((float)space.popcount(binary_code + cid * B / 64) * 2 - B);
                    (*cur_fac).factor_ip[k] = -2 / fac_norm * x_x0;
                    _max = _max > fabs(cur_fac->sqr_x[k]) ? _max : fabs(cur_fac->sqr_x[k]);
                    _max = _max > fabs(cur_fac->error[k]) ? _max : fabs(cur_fac->error[k]);
                    _max = _max > fabs(cur_fac->factor_ppc[k]) ? _max : fabs(cur_fac->factor_ppc[k]);
                    _max = _max > fabs(cur_fac->factor_ip[k]) ? _max : fabs(cur_fac->factor_ip[k]);
                    _min = _min > fabs(cur_fac->sqr_x[k]) && cur_fac->sqr_x[k] != 0 ? fabs(cur_fac->sqr_x[k]) : _min;
                    _min = _min > fabs(cur_fac->error[k]) && cur_fac->error[k] != 0 ?  fabs(cur_fac->error[k]) : _min;
                    _min = _min > fabs(cur_fac->factor_ppc[k]) && cur_fac->factor_ppc[k] != 0 ? fabs(cur_fac->factor_ppc[k]) : _min;
                    _min = _min > fabs(cur_fac->factor_ip[k]) && cur_fac->factor_ip[k] != 0 ? fabs(cur_fac->factor_ip[k]) : _min;
                } else {
                    (*cur_fac).sqr_x[k] = 0;
                    (*cur_fac).error[k] = 0;
                    (*cur_fac).factor_ppc[k] = 0;
                    (*cur_fac).factor_ip[k] = 0;
                }
            }
            cur_fac++;
        }
    }
    fac_start[C] = cur_fac;

    // compute f16 scale
    if (_max > 32768.0) {
        low_dist_scale = 32768.0 / _max;
    } else if (_min < (1.0 / 32768)) {
        low_dist_scale = (1.0 / 32768) / _min;
        low_dist_scale = std::min(low_dist_scale, (float)32768.0 / _max);
    } else {
        low_dist_scale = 1;
    }

    for (size_t i = 0; i < 1ull * N * D; ++i) {
        data_f16[i] = (float16_t)data[i];
    }
    for (size_t i = 0; i < 1ull *  C * B; ++i) {
        centroid_f16[i] = (float16_t)centroid[i];
    }
    // f32 factor quant f16
    Factor_f16 *cur_fac_f16 = fac_f16;
    const float32x4_t v_low_dist_scale = vdupq_n_f32(low_dist_scale);
    for(int i = 0; i < C; ++i) {
        int p = 0;
        fac_f16_start[i] = cur_fac_f16;
        for (Factor* cur_fac = fac_start[i]; cur_fac < fac_start[i + 1]; ++cur_fac) {
            float32x4_t v_sqr_x = vld1q_f32(cur_fac->sqr_x);
            float32x4_t v_error = vld1q_f32(cur_fac->error);
            float32x4_t v_ppc   = vld1q_f32(cur_fac->factor_ppc);
            float32x4_t v_ip    = vld1q_f32(cur_fac->factor_ip);
            v_sqr_x = vmulq_f32(v_sqr_x, v_low_dist_scale);
            v_error = vmulq_f32(v_error, v_low_dist_scale);
            v_ppc   = vmulq_f32(v_ppc  , v_low_dist_scale);
            v_ip    = vmulq_f32(v_ip   , v_low_dist_scale);
            vst1_f16(cur_fac_f16->sqr_x      + p, vcvt_f16_f32(v_sqr_x));
            vst1_f16(cur_fac_f16->error      + p, vcvt_f16_f32(v_error));
            vst1_f16(cur_fac_f16->factor_ppc + p, vcvt_f16_f32(v_ppc));
            vst1_f16(cur_fac_f16->factor_ip  + p, vcvt_f16_f32(v_ip));
            p += 4;
            if (p == 16) {
                cur_fac_f16++;
                p = 0;
            }
        }
        if (p > 0) {
            for(; p < 16; ++p){
                cur_fac_f16->sqr_x[p] = 0;
                cur_fac_f16->error[p] = 0;
                cur_fac_f16->factor_ppc[p] = 0;
                cur_fac_f16->factor_ip[p] = 0;
            }
            cur_fac_f16++;
        }
    }
}

// ==============================================================================================================================
// load impl
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::load(char * filename){
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("filename must not be null or empty");
    }

    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    uint32_t d;
    uint32_t b;
    input.read((char *) &N, sizeof(uint32_t));
    input.read((char *) &d, sizeof(uint32_t)); 
    input.read((char *) &C, sizeof(uint32_t));
    input.read((char *) &b, sizeof(uint32_t));

    if (C == 0 || C > numC) {
        throw std::runtime_error("Invalid cluster count in index file");
    }
    if (d != D || b != B) {
        throw std::runtime_error("Index header does not match template parameters");
    }

    assert(d == D);
    assert(b == B);

    FREE_DATA(centroid)
    FREE_DATA(data)
    FREE_DATA(binary_code)
    FREE_DATA(centroid_f16)
    FREE_DATA(data_f16)
    
    centroid  = static_cast<float*>(upper_bound_aligned_alloc(64, C * B * sizeof(float)));
    data  = static_cast<float*>(upper_bound_aligned_alloc(64, N * D * sizeof(float)));
    binary_code  = static_cast<uint64_t*>(upper_bound_aligned_alloc(256, N * B / 64 * sizeof(uint64_t)));
    centroid_f16  = static_cast<float16_t*>(upper_bound_aligned_alloc(64, C * B * sizeof(float16_t)));
    data_f16 = static_cast<float16_t*>(upper_bound_aligned_alloc(64, N * D * sizeof(float16_t)));

    if(start     != nullptr)    delete[] start;
    if(len       != nullptr)    delete[] len;
    if(id        != nullptr)    delete[] id;
    if(dist_to_c != nullptr)    delete[] dist_to_c;
    if(x0        != nullptr)    delete[] x0;
    if(u         != nullptr)    delete[] u;

    start        = new uint32_t [C];
    len          = new uint32_t [C];
    id           = new uint32_t [N];
    dist_to_c    = new float [N];
    x0           = new float [N];
    u            = new float [B];

    FREE_DATA(fac);
    FREE_DATA(fac_start);
    FREE_DATA(fac_f16);
    FREE_DATA(fac_f16_start);

    fac          = static_cast<Factor* >(upper_bound_aligned_alloc(64, (N / 4 + C + 1) * sizeof(Factor)));
    fac_start    = static_cast<Factor**>(upper_bound_aligned_alloc(64, (C + 1) * sizeof(Factor*)));
    fac_f16       = static_cast<Factor_f16* >(upper_bound_aligned_alloc(128, (N / 16 + C + 1) * sizeof(Factor_f16)));
    fac_f16_start = static_cast<Factor_f16**>(upper_bound_aligned_alloc(64, (C + 1) * sizeof(Factor_f16*)));

#if defined(RANDOM_QUERY_QUANTIZATION)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    for(int i=0;i<B;i++)u[i] = uniform(gen);
#else 
    for(int i=0;i<B;i++)u[i] = 0.5;
#endif 

    input.read((char *) start      , C * sizeof(uint32_t));
    input.read((char *) len        , C * sizeof(uint32_t));
    input.read((char *) id         , N * sizeof(uint32_t));
    input.read((char *) dist_to_c  , N * sizeof(float));
    input.read((char *) x0         , N * sizeof(float));

    input.read((char *) centroid   , C * B * sizeof(float));
    input.read((char *) data, 1ull * N * D * sizeof(float));
    input.read((char *) binary_code, 1ull * N * B / 64 * sizeof(uint64_t));

    pack_codes_from_file();

    compute_factor();

    std::cerr << "clean origin data" << std::endl;
    FREE_DATA(data)
    FREE_DATA(fac)
    FREE_DATA(fac_start)

    input.close();
}
// ==============================================================================================================================
// Save and Load Functions
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::save(char * filename){
    if (filename == nullptr || filename[0] == '\0') {
        throw std::invalid_argument("filename must not be null or empty");
    }

    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    uint32_t d = D;
    uint32_t b = B;
    output.write((char *) &N, sizeof(uint32_t));
    output.write((char *) &d, sizeof(uint32_t));
    output.write((char *) &C, sizeof(uint32_t));
    output.write((char *) &b, sizeof(uint32_t));

    output.write((char *) start     , C * sizeof(uint32_t));
    output.write((char *) len       , C * sizeof(uint32_t));
    output.write((char *) id        , N * sizeof(uint32_t));
    output.write((char *) dist_to_c , N * sizeof(float));
    output.write((char *) x0        , N * sizeof(float));

    output.write((char *) centroid, C * B * sizeof(float));
    output.write((char *) data, 1ull * N * D * sizeof(float));
    output.write((char *) binary_code, 1ull * N * B / 64 * sizeof(uint64_t));

    std::cerr << "Saved!" << std::endl;
    
    output.close();
}
// ==============================================================================================================================
#endif