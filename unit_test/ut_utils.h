#ifndef UT_UTILS_H_
#define UT_UTILS_H_
#include <vector>
#include <random>
#include "ivf_rabitq.h"
// #include "matrix.h"

template<typename T>
inline Matrix<T> MakeMatrix(size_t n, size_t d, T minv = T(0), T maxv = T(1)) {
    Matrix<T> m(n, d);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist((float)minv, (float)maxv);
    for (size_t i = 0; i < n * d; ++i) {
        m.data[i] = (T)dist(rng);
}
    return m;
}

template<typename T>
inline Matrix<T> MakeClusterID(size_t N, size_t C) {
    Matrix<T> m(N, 1);
    for (size_t i = 0; i < N; ++i) m.data[i] = (T)(i % C);
    return m;
}

inline Matrix<uint64_t> MakeBinary(size_t N, size_t B) {
    Matrix<uint64_t> m(N, B / 64);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N * (B / 64); ++i) m.data[i] = rng();
    return m;
}

#endif
