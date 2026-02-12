#pragma once
#include "ivf_rabitq.h"

#ifdef __aarch64__
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
    /// SOAR
    if (use_soar) {
        packed_start_spilled = new uint32_t [C];
        int cur_spilled = 0;
        for(int i=0;i<C;i++){
            packed_start_spilled[i] = cur_spilled;
            cur_spilled += (len_spilled[i] + SIZE - 1) / SIZE * SIZE * (B / 8);
        }
        packed_code_spilled = static_cast<uint8_t*>(upper_bound_aligned_alloc(32, cur_spilled * sizeof(uint8_t)));
        for(int i=0;i<C;i++){
            pack_codes<B, SIZE>(binary_code_spilled + 1ull * start_spilled[i] * (B / 64), len_spilled[i], packed_code_spilled + 1ull * packed_start_spilled[i]);
        }
    } else {
        packed_start_spilled = nullptr;
        packed_code_spilled = nullptr;
    }
#else 
    packed_start = NULL;
    packed_code  = NULL;
    packed_start_spilled = NULL;
    packed_code_spilled  = NULL;
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

    /// SOAR
    if (use_soar) {
        Factor* cur_fac_spilled = fac_spilled;
        for(int i = 0; i < C; ++i) {
            fac_start_spilled[i] = cur_fac_spilled;
            for (int j = 0; j < len[i]; j += 4) {
                for (int k = 0; k < 4; ++k) {
                    if (j + k < len[i]) {
                        int cid = start_spilled[i] + j + k;
                        long double x_x0 = (long double) dist_to_c_spilled[cid] / x0[cid];
                        (*cur_fac_spilled).sqr_x[k] = dist_to_c_spilled[cid] * dist_to_c_spilled[cid];
                        (*cur_fac_spilled).error[k] = 2 * max_x1 * std::sqrt(x_x0 * x_x0 - dist_to_c_spilled[cid] * dist_to_c_spilled[cid]);
                        (*cur_fac_spilled).factor_ppc[k] = -2 / fac_norm * x_x0  * ((float)space.popcount(binary_code_spilled + cid * B / 64) * 2 - B);
                        (*cur_fac_spilled).factor_ip[k] = -2 / fac_norm * x_x0;
                        _max = _max > fabs(cur_fac_spilled->sqr_x[k]) ? _max : fabs(cur_fac_spilled->sqr_x[k]);
                        _max = _max > fabs(cur_fac_spilled->error[k]) ? _max : fabs(cur_fac_spilled->error[k]);
                        _max = _max > fabs(cur_fac_spilled->factor_ppc[k]) ? _max : fabs(cur_fac_spilled->factor_ppc[k]);
                        _max = _max > fabs(cur_fac_spilled->factor_ip[k]) ? _max : fabs(cur_fac_spilled->factor_ip[k]);
                        _min = _min > fabs(cur_fac_spilled->sqr_x[k]) && cur_fac_spilled->sqr_x[k] != 0 ? fabs(cur_fac_spilled->sqr_x[k]) : _min;
                        _min = _min > fabs(cur_fac_spilled->error[k]) && cur_fac_spilled->error[k] != 0 ?  fabs(cur_fac_spilled->error[k]) : _min;
                        _min = _min > fabs(cur_fac_spilled->factor_ppc[k]) && cur_fac_spilled->factor_ppc[k] != 0 ? fabs(cur_fac_spilled->factor_ppc[k]) : _min;
                        _min = _min > fabs(cur_fac_spilled->factor_ip[k]) && cur_fac_spilled->factor_ip[k] != 0 ? fabs(cur_fac_spilled->factor_ip[k]) : _min;
                    } else {
                        (*cur_fac_spilled).sqr_x[k] = 0;
                        (*cur_fac_spilled).error[k] = 0;
                        (*cur_fac_spilled).factor_ppc[k] = 0;
                        (*cur_fac_spilled).factor_ip[k] = 0;
                    }
                }
                cur_fac_spilled++;
            }
        }
        fac_start_spilled[C] = cur_fac_spilled;
    }

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
    //SOAR
    if (use_soar) {
        Factor_f16* cur_fac_f16_spilled = fac_f16_spilled;
        for(int i = 0; i < C; ++i) {
            int p = 0;
            fac_f16_start_spilled[i] = cur_fac_f16_spilled;
            for (Factor* cur_fac_spilled = fac_start_spilled[i]; cur_fac_spilled < fac_start_spilled[i + 1]; ++cur_fac_spilled) {
                float32x4_t v_sqr_x = vld1q_f32(cur_fac_spilled->sqr_x);
                float32x4_t v_error = vld1q_f32(cur_fac_spilled->error);
                float32x4_t v_ppc   = vld1q_f32(cur_fac_spilled->factor_ppc);
                float32x4_t v_ip    = vld1q_f32(cur_fac_spilled->factor_ip);
                v_sqr_x = vmulq_f32(v_sqr_x, v_low_dist_scale);
                v_error = vmulq_f32(v_error, v_low_dist_scale);
                v_ppc   = vmulq_f32(v_ppc  , v_low_dist_scale);
                v_ip    = vmulq_f32(v_ip   , v_low_dist_scale);
                vst1_f16(cur_fac_f16_spilled->sqr_x      + p, vcvt_f16_f32(v_sqr_x));
                vst1_f16(cur_fac_f16_spilled->error      + p, vcvt_f16_f32(v_error));
                vst1_f16(cur_fac_f16_spilled->factor_ppc + p, vcvt_f16_f32(v_ppc));
                vst1_f16(cur_fac_f16_spilled->factor_ip  + p, vcvt_f16_f32(v_ip));
                p += 4;
                if (p == 16) {
                    cur_fac_f16_spilled++;
                    p = 0;
                }
            }
            if (p > 0) {
                for(; p < 16; ++p){
                    cur_fac_f16_spilled->sqr_x[p] = 0;
                    cur_fac_f16_spilled->error[p] = 0;
                    cur_fac_f16_spilled->factor_ppc[p] = 0;
                    cur_fac_f16_spilled->factor_ip[p] = 0;
                }
                cur_fac_f16_spilled++;
            }
        }
        for (size_t i = 0; i < 1ull * N * D; ++i) {
            data_f16_spilled[i] = (float16_t)data_spilled[i];
        }
    }

}

#define FREE_DATA(VARIABLE)     \
    if (VARIABLE != NULL) {     \
        std::free(VARIABLE);    \
        VARIABLE = nullptr;     \
    }

// =================================================== load function ===========================================================================
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::load(char * filename){
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    uint32_t d;
    uint32_t b;
    input.read((char *) &N, sizeof(uint32_t));
    input.read((char *) &d, sizeof(uint32_t)); 
    input.read((char *) &C, sizeof(uint32_t));
    input.read((char *) &b, sizeof(uint32_t));

    std::cerr << d << std::endl;
    assert(d == D);
    assert(b == B);

    if (u != NULL) { delete [] u; u = NULL; }
    u = new float [B];
#if defined(RANDOM_QUERY_QUANTIZATION)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    for(int i=0;i<B;i++)u[i] = uniform(gen);
#else 
    for(int i=0;i<B;i++)u[i] = 0.5;
#endif 
    FREE_DATA(centroid)
    FREE_DATA(data)
    FREE_DATA(binary_code)
    
    // Free previously allocated arrays before reallocating
    if(start != NULL)        delete [] start;
    if(len != NULL)          delete [] len;
    if(id != NULL)           delete [] id;
    if(dist_to_c != NULL)    delete [] dist_to_c;
    if(x0 != NULL)           delete [] x0;

    centroid  = static_cast<float*>(upper_bound_aligned_alloc(64, C * B * sizeof(float)));
    data  = static_cast<float*>(upper_bound_aligned_alloc(64, N * D * sizeof(float)));
    binary_code  = static_cast<uint64_t*>(upper_bound_aligned_alloc(256, N * B / 64 * sizeof(uint64_t)));

    centroid_f16  = static_cast<float16_t*>(upper_bound_aligned_alloc(64, C * B * sizeof(float16_t)));
    data_f16 = static_cast<float16_t*>(upper_bound_aligned_alloc(64, N * D * sizeof(float16_t)));

    start        = new uint32_t [C];
    len          = new uint32_t [C];
    id           = new uint32_t [N];
    dist_to_c    = new float [N];
    x0           = new float [N];

    // Free previously allocated factor arrays before reallocating
    FREE_DATA(fac)
    FREE_DATA(fac_start)
    FREE_DATA(fac_f16)
    FREE_DATA(fac_f16_start)
    if(packed_start != NULL)            delete [] packed_start;
    packed_start = nullptr;
    
    fac          = static_cast<Factor* >(upper_bound_aligned_alloc(64, (N / 4 + C + 1) * sizeof(Factor)));
    fac_start    = static_cast<Factor**>(upper_bound_aligned_alloc(64, (C + 1) * sizeof(Factor*)));
    fac_f16       = static_cast<Factor_f16* >(upper_bound_aligned_alloc(128, (N / 16 + C + 1) * sizeof(Factor_f16)));
    fac_f16_start = static_cast<Factor_f16**>(upper_bound_aligned_alloc(64, (C + 1) * sizeof(Factor_f16*)));

    input.read((char *) start      , C * sizeof(uint32_t));
    input.read((char *) len        , C * sizeof(uint32_t));
    input.read((char *) id         , N * sizeof(uint32_t));
    input.read((char *) dist_to_c  , N * sizeof(float));
    input.read((char *) x0         , N * sizeof(float));

    input.read((char *) centroid   , C * B * sizeof(float));
    input.read((char *) data, 1ull * N * D * sizeof(float));
    input.read((char *) binary_code, 1ull * N * B / 64 * sizeof(uint64_t));

    /// SOAR
    if (!input.eof() && use_soar) {
        FREE_DATA(data_spilled)
        FREE_DATA(binary_code_spilled)
        // Free previously allocated SOAR arrays before reallocating
        if(start_spilled != NULL)        delete [] start_spilled;
        if(len_spilled != NULL)          delete [] len_spilled;
        if(id_spilled != NULL)           delete [] id_spilled;
        if(dist_to_c_spilled != NULL)   delete [] dist_to_c_spilled;
        if(x0_spilled != NULL)          delete [] x0_spilled;
        
        data_spilled  = static_cast<float*>(upper_bound_aligned_alloc(64, N * D * sizeof(float)));
        data_f16_spilled = static_cast<float16_t*>(upper_bound_aligned_alloc(64, N * D * sizeof(float16_t)));
        binary_code_spilled  = static_cast<uint64_t*>(upper_bound_aligned_alloc(256, N * B / 64 * sizeof(uint64_t)));
        start_spilled        = new uint32_t [C];
        len_spilled          = new uint32_t [C];
        id_spilled           = new uint32_t [N];
        dist_to_c_spilled    = new float [N];
        x0_spilled           = new float [N];
        // Free previously allocated SOAR factor arrays before reallocating
        FREE_DATA(fac_spilled)
        FREE_DATA(fac_start_spilled)
        FREE_DATA(fac_f16_spilled)
        FREE_DATA(fac_f16_start_spilled)
        if(packed_start_spilled != NULL)    delete [] packed_start_spilled;
        packed_start_spilled = nullptr;
        
        fac_spilled          = static_cast<Factor* >(upper_bound_aligned_alloc(64, (N / 4 + C + 1) * sizeof(Factor)));
        fac_start_spilled    = static_cast<Factor**>(upper_bound_aligned_alloc(64, (C + 1) * sizeof(Factor*)));
        fac_f16_spilled          = static_cast<Factor_f16* >(upper_bound_aligned_alloc(128, (N / 16 + C + 1) * sizeof(Factor_f16)));
        fac_f16_start_spilled    = static_cast<Factor_f16**>(upper_bound_aligned_alloc(64, (C + 1) * sizeof(Factor_f16*)));

        input.read((char *) start_spilled      , C * sizeof(uint32_t));
        input.read((char *) len_spilled        , C * sizeof(uint32_t));
        input.read((char *) id_spilled         , N * sizeof(uint32_t));
        input.read((char *) dist_to_c_spilled  , N * sizeof(float));
        input.read((char *) x0_spilled         , N * sizeof(float));

        input.read((char *) data_spilled, 1ull * N * D * sizeof(float));
        input.read((char *) binary_code_spilled, 1ull * N * B / 64 * sizeof(uint64_t));
    } else {
        // Free SOAR arrays if switching from SOAR to non-SOAR mode
        if(start_spilled != NULL)        delete [] start_spilled;
        if(len_spilled != NULL)          delete [] len_spilled;
        if(id_spilled != NULL)           delete [] id_spilled;
        if(dist_to_c_spilled != NULL)    delete [] dist_to_c_spilled;
        if(x0_spilled != NULL)          delete [] x0_spilled;
        start_spilled = nullptr;
        len_spilled = nullptr;
        id_spilled = nullptr;
        dist_to_c_spilled = nullptr;
        x0_spilled = nullptr;
        FREE_DATA(data_spilled)
        FREE_DATA(binary_code_spilled)
        FREE_DATA(data_f16_spilled)
        FREE_DATA(fac_spilled)
        FREE_DATA(fac_start_spilled)
        FREE_DATA(fac_f16_spilled)
        FREE_DATA(fac_f16_start_spilled)
        use_soar = false;
    }

    pack_codes_from_file();

    compute_factor();

    std::cerr << "clean load_soar origin data" << std::endl;
    FREE_DATA(data)
    FREE_DATA(data_spilled)
    FREE_DATA(fac)
    FREE_DATA(fac_start)
    FREE_DATA(fac_spilled)
    FREE_DATA(fac_start_spilled)

    input.close();
}

// ========================================================= Save Function =====================================================================
//
template <uint32_t D, uint32_t B>
void IVFRN<D, B>::save(char * filename){
    std::ofstream output(filename, std::ios::binary);

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

    if (use_soar) {
        output.write((char *) start_spilled     , C * sizeof(uint32_t));
        output.write((char *) len_spilled       , C * sizeof(uint32_t));
        output.write((char *) id_spilled        , N * sizeof(uint32_t));
        output.write((char *) dist_to_c_spilled , N * sizeof(float));
        output.write((char *) x0_spilled        , N * sizeof(float));
        output.write((char *) data_spilled, 1ull * N * D * sizeof(float));
        output.write((char *) binary_code_spilled, 1ull * N * B / 64 * sizeof(uint64_t));
        std::cerr << "Saved SOAR!" << std::endl;
    } else {
        std::cerr << "Saved!" << std::endl;
    }
    output.close();
}
// =============================================================================================================================================
#endif