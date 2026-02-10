#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdint>

// #include "ivf_rabitQ.h"
#include "ut_utils.h"


// ======================== Search No SOAR测试 ========================

TEST(IVFRNSearch, HighCNoSoarRecall) {
    constexpr uint32_t D = 127;
    constexpr uint32_t B = 128;
    const size_t N = 10000;
    const size_t C = 64;
    const size_t QN = 10;
    const size_t K = 10;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    IVFRN<D, B> index0(
        X, centroids, dist_to_centroid, x0,
        cluster_id, binary,
        nullptr, nullptr, nullptr, nullptr
    );

    const char* fname = "./data/rabitq_test_search.bin";
    index0.save((char*)fname);

    IVFRN<D, B> index;
    index.load((char*)fname);

    auto Q = MakeMatrix<float>(QN, D);

    float avg_recall = 0.0f;

    for (size_t qi = 0; qi < QN; ++qi) {
        float* q = Q.data + qi * D;

        std::vector<float> rq(B, 0.1f);

        auto pq = index.search(
            q,
            rq.data(),
            K,
            /*nprobe=*/C,
            /*soar_lambda=*/0.0f
        );

        EXPECT_GE(pq.size(), K);
    }

}

TEST(IVFRNSearch, LowCNoSoarRecall) {
    constexpr uint32_t D = 128;
    constexpr uint32_t B = 128;
    const size_t N = 10000;
    const size_t C = 4;
    const size_t QN = 10;
    const size_t K = 10;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    IVFRN<D, B> index0(
        X, centroids, dist_to_centroid, x0,
        cluster_id, binary,
        nullptr, nullptr, nullptr, nullptr
    );

    const char* fname = "./data/rabitq_test_search.bin";
    index0.save((char*)fname);

    IVFRN<D, B> index;
    index.load((char*)fname);

    auto Q = MakeMatrix<float>(QN, D);

    float avg_recall = 0.0f;

    for (size_t qi = 0; qi < QN; ++qi) {
        float* q = Q.data + qi * D;

        std::vector<float> rq(B, 0.1f);

        auto pq = index.search(
            q,
            rq.data(),
            K,
            /*nprobe=*/C,
            /*soar_lambda=*/0.0f
        );

        EXPECT_GE(pq.size(), K);
    }

}

// ======================== Search SOAR 分支测试 ========================

TEST(IVFRNSearch, HighCSoarRecall) {
    constexpr uint32_t D = 128;
    constexpr uint32_t B = 128;
    const size_t N = 10000;
    const size_t C = 64;
    const size_t QN = 10;
    const size_t K = 10;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    auto dist_to_spilled = MakeMatrix<float>(N, 1);
    auto x0_spilled = MakeMatrix<float>(N, 1);
    auto spilled_labels = MakeClusterID<uint32_t>(N, C);
    auto binary_spilled = MakeBinary(N, B);

    IVFRN<D, B> index0(
        X, centroids, dist_to_centroid, x0,
        cluster_id, binary,
        &dist_to_spilled, &x0_spilled, &spilled_labels, &binary_spilled
    );

    const char* fname = "./data/rabitq_test_search_soar.bin";
    // Constructor automatically sets use_soar = true when SOAR parameters are provided
    index0.save((char*)fname);

    IVFRN<D, B> index;
    index.use_soar = true;  // Set flag before loading SOAR data
    index.load((char*)fname);

    auto Q = MakeMatrix<float>(QN, D);

    for (size_t qi = 0; qi < QN; ++qi) {
        float* q = Q.data + qi * D;

        std::vector<float> rq(B, 0.1f);

        auto pq = index.search(
            q,
            rq.data(),
            K,
            /*nprobe=*/C,
            /*soar_lambda=*/1.0f
        );
        EXPECT_GE(pq.size(), K);
    }
}

TEST(IVFRNSearch, LowCSoarRecall) {
    constexpr uint32_t D = 128;
    constexpr uint32_t B = 128;
    const size_t N = 10000;
    const size_t C = 4;
    const size_t QN = 10;
    const size_t K = 10;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    auto dist_to_spilled = MakeMatrix<float>(N, 1);
    auto x0_spilled = MakeMatrix<float>(N, 1);
    auto spilled_labels = MakeClusterID<uint32_t>(N, C);
    auto binary_spilled = MakeBinary(N, B);

    IVFRN<D, B> index0(
        X, centroids, dist_to_centroid, x0,
        cluster_id, binary,
        &dist_to_spilled, &x0_spilled, &spilled_labels, &binary_spilled
    );

    const char* fname = "./data/rabitq_test_search_soar.bin";
    // Constructor automatically sets use_soar = true when SOAR parameters are provided
    index0.save((char*)fname);

    IVFRN<D, B> index;
    index.use_soar = true;  // Set flag before loading SOAR data
    index.load((char*)fname);

    auto Q = MakeMatrix<float>(QN, D);

    for (size_t qi = 0; qi < QN; ++qi) {
        float* q = Q.data + qi * D;

        std::vector<float> rq(B, 0.1f);

        auto pq = index.search(
            q,
            rq.data(),
            K,
            /*nprobe=*/C,
            /*soar_lambda=*/1.0f
        );
        EXPECT_GE(pq.size(), K);
    }
}

/* ======================== nprobe 行为测试 ======================== */
TEST(IVFRNTest, DifferentPredNprobeBranch) {    
    constexpr uint32_t D = 127;
    constexpr uint32_t B = 128;
    const size_t N = 5120;
    const size_t C = 63;
    const size_t K = 10;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    IVFRN<D, B> index0(
        X, centroids, dist_to_centroid, x0,
        cluster_id, binary,
        nullptr, nullptr, nullptr, nullptr
    );

    const char* fname = "./data/rabitq_test_pre_nprobe.bin";
    index0.save((char*)fname);

    IVFRN<D, B> index;
    index.load((char*)fname);

    std::vector<float> query(D, 1.0f);
    std::vector<float> rd_query(B, 1.0f);

    auto pred_low = [](Entry*, int, double* res) {
        *res = 0.0;
    };

    /* =============== 不进入pre_nprobe分支 =============== */
    auto r1 = index.search(
        query.data(), rd_query.data(),
        K,
        /*nprobe=*/20,
        /*soar_lambda=*/0,
        /*threshold=*/0.5,
        /*pred_nprobe=*/-1,
        pred_low
    );
    EXPECT_GT(r1.size(), 0);

    /* =============== pre_nprobe = 0 不进入分支 =============== */
    auto r2 = index.search(
        query.data(), rd_query.data(),
        K,
        /*nprobe=*/20,
        /*soar_lambda=*/0,
        /*threshold=*/0.5,
        /*pred_nprobe=*/0,
        pred_low
    );
    EXPECT_GT(r2.size(), 0);

    /* =============== 进入pre_nprobe分支且 <= threshold 分支 =============== */
    auto r3 = index.search(
        query.data(), rd_query.data(),
        K,
        /*nprobe=*/20,
        /*soar_lambda=*/0,
        /*threshold=*/0.5,
        /*pred_nprobe=*/5,
        pred_low
    );
    EXPECT_GT(r3.size(), 0);

    auto pred_high = [](Entry*, int, double* res) {
        *res = 1000.0;
    };

    /* =============== 进入pre_nprobe分支且 > threshold 分支 =============== */
    auto r4 = index.search(
        query.data(), rd_query.data(),
        K,
        /*nprobe=*/20,
        /*soar_lambda=*/0,
        /*threshold=*/0.5,
        /*pred_nprobe=*/5,
        pred_high
    );
    EXPECT_GT(r4.size(), 0);
}

// ======================== Save NOSOAR 分支测试 ========================

TEST(IVFRNLoadSave, Basic) {
    constexpr uint32_t D = 8;
    constexpr uint32_t B = 64;
    const size_t N = 32;
    const size_t C = 7;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    IVFRN<D, B> index(X, centroids, dist_to_centroid, x0, cluster_id, binary,
                      nullptr, nullptr, nullptr, nullptr);

    const char* fname = "./data/rabitq_test_ls.bin";
    index.save((char*)fname);

    IVFRN<D, B> loaded;
    loaded.load((char*)fname);

    EXPECT_EQ(loaded.N, N);
    EXPECT_EQ(loaded.C, C);
    EXPECT_FALSE(loaded.use_soar);

    for (size_t i = 0; i < C * B; ++i) {
        EXPECT_FLOAT_EQ(loaded.centroid[i], centroids.data[i]);
    }

    for (size_t i = 0; i < N; ++i) {
        uint32_t orig = loaded.id[i];
        EXPECT_FLOAT_EQ(loaded.dist_to_c[i], dist_to_centroid.data[orig]);
        EXPECT_FLOAT_EQ(loaded.x0[i], x0.data[orig]);

        for (size_t j = 0; j < D; ++j) {
            EXPECT_EQ(loaded.data_f16[i * D + j], (float16_t)(X.data[orig * D + j]));
        }

        EXPECT_EQ(
            std::memcmp(loaded.binary_code + i * (B/64), binary.data + orig * (B/64), (B/64) * sizeof(uint64_t)),
            0
        );
    }

    std::remove(fname);
}

/* ===================== 2. Save / Load SOAR测试 ===================== */
TEST(IVFRNLoadSave, SOAR) {
    constexpr uint32_t D = 8;
    constexpr uint32_t B = 64;
    const size_t N = 32;
    const size_t C = 7;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    auto dist_to_spilled = MakeMatrix<float>(N, 1);
    auto x0_spilled = MakeMatrix<float>(N, 1);
    auto spilled_labels = MakeClusterID<uint32_t>(N, C);
    auto binary_spilled = MakeBinary(N, B);

    IVFRN<D, B> index(X, centroids, dist_to_centroid, x0,
                      cluster_id, binary,
                      &dist_to_spilled, &x0_spilled, &spilled_labels, &binary_spilled);

    const char* fname = "./data/rabitq_test_ls_soar.bin";
    index.save((char*)fname);

    IVFRN<D, B> loaded;
    loaded.use_soar = true;
    loaded.load((char*)fname);

    EXPECT_EQ(loaded.N, N);
    EXPECT_EQ(loaded.C, C);
    EXPECT_TRUE(loaded.use_soar);

    for (size_t i = 0; i < C * B; ++i) {
        EXPECT_FLOAT_EQ(loaded.centroid[i], centroids.data[i]);
    }

    for (size_t i = 0; i < N; ++i) {
        uint32_t orig = loaded.id_spilled[i];
        EXPECT_FLOAT_EQ(loaded.dist_to_c_spilled[i], dist_to_spilled.data[orig]);
        EXPECT_FLOAT_EQ(loaded.x0_spilled[i], x0_spilled.data[orig]);

        for (size_t j = 0; j < D; ++j) {
            EXPECT_EQ(loaded.data_f16[i * D + j], (float16_t)(X.data[orig * D + j]));
        }

        EXPECT_EQ(
            std::memcmp(loaded.binary_code_spilled + i * (B/64),
                        binary_spilled.data + orig * (B/64),
                        (B/64) * sizeof(uint64_t)),
            0
        );
    }

    std::remove(fname);
}


/* ===============1、无参数构造函数测试=============== */
TEST(IVFRNConstruction, DefaultConstructor) {
    constexpr uint32_t D = 8;
    constexpr uint32_t B = 64;
    IVFRN<D, B> index;

    EXPECT_EQ(index.N, 0u);
    EXPECT_EQ(index.C, 0u);

    EXPECT_EQ(index.start, nullptr);
    EXPECT_EQ(index.len, nullptr);
    EXPECT_EQ(index.id, nullptr);

    EXPECT_EQ(index.data, nullptr);
    EXPECT_EQ(index.binary_code, nullptr);
    EXPECT_EQ(index.centroid, nullptr);

    EXPECT_EQ(index.x0, nullptr);
    EXPECT_EQ(index.dist_to_c, nullptr);
    EXPECT_EQ(index.u, nullptr);
    EXPECT_EQ(index.fac, nullptr);
    EXPECT_EQ(index.fac_start, nullptr);

    EXPECT_EQ(index.centroid_f16, nullptr);
    EXPECT_EQ(index.data_f16, nullptr);

    EXPECT_EQ(index.start_spilled, nullptr);
    EXPECT_EQ(index.len_spilled, nullptr);
    EXPECT_EQ(index.id_spilled, nullptr);
    EXPECT_EQ(index.x0_spilled, nullptr);
    EXPECT_EQ(index.dist_to_c_spilled, nullptr);
    EXPECT_EQ(index.data_spilled, nullptr);
    EXPECT_EQ(index.binary_code_spilled, nullptr);
    EXPECT_EQ(index.data_f16_spilled, nullptr);

    EXPECT_FALSE(index.use_soar);
}

/* ===============2、带参数构造函数测试（无 SOAR）=============== */
TEST(IVFRNConstruction, NormalConstructorNoSOAR) {
    constexpr uint32_t D = 8;
    constexpr uint32_t B = 64;
    const size_t N = 32;
    const size_t C = 7;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    IVFRN<D, B> index(
        X, centroids, dist_to_centroid, x0,
        cluster_id, binary,
        nullptr, nullptr, nullptr, nullptr
    );

    EXPECT_EQ(index.N, N);
    EXPECT_EQ(index.C, C);
    EXPECT_FALSE(index.use_soar);

    EXPECT_NE(index.start, nullptr);
    EXPECT_NE(index.len, nullptr);
    EXPECT_NE(index.id, nullptr);
    EXPECT_NE(index.data, nullptr);
    EXPECT_NE(index.binary_code, nullptr);
    EXPECT_NE(index.centroid, nullptr);

    // --- sum 正确性 ---
    size_t sum = 0;
    for (size_t i = 0; i < C; ++i) {
        sum += index.len[i];
    }
    EXPECT_EQ(sum, N);

    // --- id 正确性 ---
    std::vector<bool> used(N, false);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_LT(index.id[i], N);
        EXPECT_FALSE(used[index.id[i]]);
        used[index.id[i]] = true;
    }

    // --- cluster bucket 正确性 ---
    for (size_t c = 0; c < C; ++c) {
        for (size_t i = index.start[c]; i < index.start[c] + index.len[c]; ++i) {
            uint32_t original = index.id[i];
            EXPECT_EQ(cluster_id.data[original], c);
            EXPECT_FLOAT_EQ(index.dist_to_c[i], dist_to_centroid.data[original]);
            EXPECT_FLOAT_EQ(index.x0[i], x0.data[original]);
        }
    }

    // --- centroid 正确性 ---
    for (size_t i = 0; i < C * B; ++i) {
        EXPECT_FLOAT_EQ(index.centroid[i], centroids.data[i]);
    }

    // --- data / binary 正确性 ---
    for (size_t i = 0; i < N; ++i) {
        uint32_t original = index.id[i];

        EXPECT_EQ(std::memcmp(index.data + i * D, X.data + original * D, D * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(index.binary_code + i * (B / 64), binary.data + original * (B / 64),
                              (B / 64) * sizeof(uint64_t)), 0);
    }      
}

/* ===============3、带参数构造函数测试（带 SOAR）=============== */
TEST(IVFRNConstruction, ConstructorWithSOAR) {
    constexpr uint32_t D = 8;
    constexpr uint32_t B = 64;
    const size_t N = 32;
    const size_t C = 7;

    auto X = MakeMatrix<float>(N, D);
    auto centroids = MakeMatrix<float>(C, B);
    auto dist_to_centroid = MakeMatrix<float>(N, 1);
    auto x0 = MakeMatrix<float>(N, 1);
    auto cluster_id = MakeClusterID<uint32_t>(N, C);
    auto binary = MakeBinary(N, B);

    auto dist_to_spilled = MakeMatrix<float>(N, 1);
    auto x0_spilled = MakeMatrix<float>(N, 1);
    auto spilled_labels = MakeClusterID<uint32_t>(N, C);
    auto binary_spilled = MakeBinary(N, B);

    IVFRN<D, B> index(
        X, centroids, dist_to_centroid, x0,
        cluster_id, binary,
        &dist_to_spilled, &x0_spilled, &spilled_labels, &binary_spilled
    );

    EXPECT_EQ(index.N, N);
    EXPECT_EQ(index.C, C);
    EXPECT_TRUE(index.use_soar);

    EXPECT_NE(index.start_spilled, nullptr);
    EXPECT_NE(index.len_spilled, nullptr);
    EXPECT_NE(index.id_spilled, nullptr);
    EXPECT_NE(index.data_spilled, nullptr);
    EXPECT_NE(index.binary_code_spilled, nullptr);

    size_t sum = 0;
    for (size_t i = 0; i < C; ++i) {
        sum += index.len_spilled[i];
    }
    EXPECT_EQ(sum, N);

    std::vector<bool> used(N, false);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_LT(index.id_spilled[i], N);
        EXPECT_FALSE(used[index.id_spilled[i]]);
        used[index.id_spilled[i]] = true;
    }

    for (size_t c = 0; c < C; ++c) {
        for (size_t i = index.start_spilled[c]; i < index.start_spilled[c] + index.len_spilled[c]; ++i) {
            uint32_t original = index.id_spilled[i];
            EXPECT_EQ(spilled_labels.data[original], c);
            EXPECT_FLOAT_EQ(index.dist_to_c_spilled[i], dist_to_spilled.data[original]);
            EXPECT_FLOAT_EQ(index.x0_spilled[i], x0_spilled.data[original]);
        }
    }

    for (size_t i = 0; i < N; ++i) {
        uint32_t original = index.id_spilled[i];

        EXPECT_EQ(std::memcmp(index.data_spilled + i * D, X.data + original * D, D * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(index.binary_code_spilled + i * (B / 64),
                              binary_spilled.data + original * (B / 64),
                              (B / 64) * sizeof(uint64_t)), 0);
    }
}

/* ===============4、矩阵计算测试=============== */
TEST(IVFRNTest, MatrixTest) {
    constexpr uint32_t D = 8;
    constexpr uint32_t N = 16;

    auto X = MakeMatrix<float>(N, D);
    auto Q = MakeMatrix<float>(N, D);
    std::memset(Q.data, 0, sizeof(float) * N * D);

    float dis = X.dist(0, Q, 0);
    float norm = normalize(X.data, D);

    EXPECT_FLOAT_EQ(dis, norm * norm);
}

/* ===============4、其他测试=============== */
TEST(IVFRNTest, OtherTest) {
    constexpr uint64_t x = 0x5555555555555555;
    print_binary(x);
    print_binary((uint8_t)x);

    constexpr uint32_t N = 64;
    constexpr uint32_t D = 16;
    constexpr uint32_t K = 10;
    struct rusage startTime, endTime;
    float userTime, sysTime;

    GetCurTime(&startTime);

    auto X = MakeMatrix<float>(N, D);
    auto Q = MakeMatrix<float>(N, D);
    auto G = MakeClusterID<unsigned>(N * K, N);

    ResultHeap gt = getGroundtruth(X, Q, 0, G.data, K);

    float ratio = getRatio(0, Q, X, G, gt);

    int recall = getRecall(gt, gt);

    EXPECT_EQ(recall, gt.size());

    size_t curmemory = getCurrentRSS();

    size_t totalmemory = getPeakRSS();

    EXPECT_GE(totalmemory, curmemory);

    normalizeVector(X.data, N * D);
    
    GetCurTime(&endTime);

    GetTime(&startTime, &endTime, &userTime, &sysTime);

    EXPECT_GT(userTime, 0);
    EXPECT_GT(sysTime, 0);
}