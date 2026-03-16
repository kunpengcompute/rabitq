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
#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <iostream>
#include <fstream>
#include <cstdio>

#include <ctime>
#include <cmath>
#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"
#include <getopt.h>

#include <omp.h>    
#include <thread>

using namespace std;

template<uint32_t D, uint32_t B>
void findTruth(std::vector<int64_t> &approximateGT, float *train_data, float *rand_train_data, int n, int n_leaves, const IVFRN<D, B> &ivf, int k, const char* dataset, float soar_lambda) {
    std::cerr << "in findTruth n_leaves = " << n_leaves << std::endl;
    std::cerr << "in findTruth D = " << D << "  BB = " << B << std::endl;
    for(int i=0;i<n;i++){
        // std::cerr << "start in i = " << i << std::endl;
        ResultHeap KNNs = ivf.search(train_data + i * D, rand_train_data + i * B, k, n_leaves, soar_lambda);
        // std::cerr << "finish i  = " << i << " find truth\n";
        while(KNNs.empty() == false){
            approximateGT.push_back(static_cast<int64_t>(KNNs.top().second));
            KNNs.pop();
}
    }
    
    char path[512];
    std::snprintf(path, sizeof(path),
                  "data/%s/approximateGT.bin", dataset);


    std::ofstream of2(path  , std::ios::out | std::ios::binary);
    of2.write(reinterpret_cast<char *>(approximateGT.data()), approximateGT.size() * sizeof(int64_t));
    of2.close();

}

template<uint32_t D, uint32_t B>
void train(const Matrix<float> &Q, const Matrix<float> &RandQ,
            const IVFRN<D, B> &ivf, int k, std::vector<int64_t> base_labels, int nProbeMax, int qsize, const char* dataset, float soar_lambda){
    if (k <= 0) {
        throw std::invalid_argument("train: k must be greater than 0");
    }
    float sys_t, usr_t, usr_t_sum = 0, total_time=0, search_time=0;
    struct rusage run_start, run_end;
    const float maxRecall = 0.9999;
    
    // const Matrix<float> prob_info;  // Unused variable, commented out
    std::vector<int64_t> expectedNprobe(qsize);

    // const int num_threads = omp_get_max_threads();
    // std::vector<int> cpu_ids = getAvailableCPUs();

    // pthread_mutex_t mtx;
    // pthread_cond_t cond;
    // bool ready = false;
    // pthread_mutex_init(&mtx, NULL);
    // pthread_cond_init(&cond, NULL);


    for (int i = 0; i < qsize; ++i){
        // std::cerr << "star in find nprobe i = " << i << std::endl;
        int npMin = 1;
        int npMax = nProbeMax;
        int curNprobe = (npMin + npMax) / 2;
        while (npMin <= npMax) {
            curNprobe = (npMin + npMax) / 2;

            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, RandQ.data + i * RandQ.d, k, curNprobe, soar_lambda);
            int tmp_correct = 0;
            while(KNNs.empty() == false){
                int id = KNNs.top().second;
                KNNs.pop();
                for(int j=0;j<k;j++)
                    if(id == base_labels[i * k + j])tmp_correct ++;
            }
            float recall = 1.0f * tmp_correct / (k);
            if (recall < maxRecall) {
                npMin = curNprobe + 1;
            } else if (recall > maxRecall) {
                npMax = curNprobe - 1;
            } else {
                expectedNprobe[i] = curNprobe;
                break;  // Found exact match, exit loop
            }
        }
        if (expectedNprobe[i] == 0) {
            expectedNprobe[i] = npMax + 1;
        }
    }
    
    char path[512];
    std::snprintf(path, sizeof(path),
                  "data/%s/expectedNprobe1.bin", dataset);


    std::ofstream of2(path, std::ios::out | std::ios::binary);
    of2.write(reinterpret_cast<char *>(expectedNprobe.data()), expectedNprobe.size()*sizeof(int64_t));
    of2.close();

}

bool readBinaryFile(const std::string& filename, std::vector<int64_t>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t numElements = fileSize / sizeof(int64_t);

    data.resize(numElements);

    if (file.read(reinterpret_cast<char*>(data.data()), fileSize)) {
        std::cout << "成功读取 " << numElements << " 个 int64_t 数据" << std::endl;
    } else {
        std::cerr << "读取文件时出错" << std::endl;
        return false;
    }

    file.close();
    return true;
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"K",                           required_argument, 0, 'k'},
        {"Nprobe",                      required_argument, 0, 'n'},

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'd'},
        {"source",                      required_argument, 0, 's'},
        {"result_path",                 required_argument, 0, 'r'},

        {"data_path",                   required_argument, 0, 'p'},
        {"metric_type",                      required_argument, 0, 'm'},
        {"soar_lambda",                      required_argument, 0, 'a'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char result_path[256] = "";
    char data_path[256] = ""; 
    char metric_type[256] = "";
    char scan_type[256] = "";
    
#if defined(FAST_SCAN)
    strcpy(scan_type, "fastscan");
#elif defined(SCAN)
    char result_file_view[256] = "";
    strcpy(scan_type, "scan");
#endif
    int topk = 10;
    int n_leaves = 0;
    float soar_lambda = 0;
    
    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:r:k:n:s:p:m:a:", longopts, &ind);
        switch (iarg){
            case 'k':
                if(optarg){
                    topk = atoi(optarg);
                    std::cout << "TopK set to: " << topk << std::endl;
                } else {
                    std::cerr << "Warning: -k option requires an argument, using default value 10" << std::endl;
                }
                break;
            case 'n':
                if(optarg){
                    n_leaves = atoi(optarg);
                    std::cout << "n_leaves set to: " << n_leaves << std::endl;
                } else {
                    std::cerr << "Warning: -n option requires an argument, using default value 0" << std::endl;
                }
                break;           
            case 's':
                if(optarg){
                    strncpy(source, optarg, sizeof(source) - 1);
                    source[sizeof(source) - 1] = '\0';
                }
                break;
            case 'r':
                if(optarg){
                    strncpy(result_path, optarg, sizeof(result_path) - 1);
                    result_path[sizeof(result_path) - 1] = '\0';
                }
                break;
            case 'd':
                if(optarg){
                    strncpy(dataset, optarg, sizeof(dataset) - 1);
                    dataset[sizeof(dataset) - 1] = '\0';
                }
                break;
            case 'p':
                if(optarg){
                    strncpy(data_path, optarg, sizeof(data_path) - 1);
                    data_path[sizeof(data_path) - 1] = '\0';
                }
                break;
            case 'm':
                if(optarg){
                    strncpy(metric_type, optarg, sizeof(metric_type) - 1);
                    metric_type[sizeof(metric_type) - 1] = '\0';
                }
                break;           
            case 'a':
                if(optarg){
                    soar_lambda = atof(optarg);
                    std::cout << "soar_lambda set to: " << soar_lambda << std::endl;
                } else {
                    std::cerr << "Warning: -a option requires an argument, using default value 0" << std::endl;
                }
                break; 
        }
    }

    if (topk <= 0) {
        std::cerr << "Error: -k must be greater than 0" << std::endl;
        return 1;
    }
    
    // ================================================================================================================================
    // Data Files
    float *xb_;
    float *xq_;
    int64_t *gt_ids_;
    float *gt_dists_;
    int32_t nb_, nq_, dim_, gt_closest;

    loadHDF(data_path, nb_, nq_, dim_, gt_closest, xb_, xq_, gt_ids_, gt_dists_, metric_type);
    delete[] gt_dists_;
    gt_dists_ = nullptr;

    Matrix<float> Q(xq_, nq_, dim_, false);
    Matrix<float> X(xb_, nb_, dim_, false);
    Matrix<int64_t> G(gt_ids_, nq_, gt_closest, false);
    delete[] xq_;
    xq_ = nullptr;
    delete[] xb_;
    xb_ = nullptr;
    delete[] gt_ids_;
    gt_ids_ = nullptr;

    char transformation_path[256] = "";
    snprintf(transformation_path, sizeof(transformation_path), "%sP_C%d_B%d.fvecs", source, numC, BB);
    Matrix<float> P(transformation_path);

    char index_path[256] = "";
    snprintf(index_path, sizeof(index_path), "%sivfrabitq_%s_%d_B%d.index", source, scan_type, numC, BB);
    std::cerr << index_path << std::endl;
#if defined(FAST_SCAN)
    char result_file_view[256] = "";
    snprintf(result_file_view, sizeof(result_file_view), "%s%s_ivfrabitq%d_B%d_fast_scan.log", result_path, dataset, numC, BB);
#elif defined(SCAN)
    char result_file_view[256] = "";
    snprintf(result_file_view, sizeof(result_file_view), "%s%s_ivfrabitq%d_B%d_scan.log", result_path, dataset, numC, BB);
#endif
    std::cerr << "Loading Succeed!" << std::endl;
    // ================================================================================================================================

    if (freopen(result_file_view, "a", stdout) == nullptr) {
        std::cerr << "Error: failed to open result file: " << result_file_view << std::endl;
        return 1;
    }
    
    IVFRN<DIM, BB> ivf;
    if (soar_lambda > 0){
        ivf.use_soar = true;  // Set flag before loading SOAR data
        ivf.load(index_path);
    } else {
        ivf.load(index_path);
    }
    Matrix<float> RandX(X.n, BB, X);
    RandX = mul(RandX, P);

    /// FIND TRUTH
    int dim = dim_;
    std::cerr << "dim = " << dim << " random dim = " << RandX.d <<std::endl;
    int qsize = X.n < 50000 ? X.n : 50000;
    
    std::vector<float> train_data(qsize * dim);
    std::copy(X.data, X.data + qsize * dim, train_data.begin());
    std::vector<float> rand_train_data(qsize * RandX.d);
    std::copy(RandX.data, RandX.data + qsize * RandX.d, rand_train_data.begin());

    std::cerr << "--> findTruth dataset = " << dataset << std::endl;

    std::vector<int64_t> approximateGT;
    findTruth(approximateGT, train_data.data(), rand_train_data.data(), qsize, n_leaves, ivf, topk, dataset, soar_lambda);

    
    ///// SEARCH EXPECT NPROBE
    // std::vector<int64_t> approximateGT;
    // if (readBinaryFile("approximateGT.bin", approximateGT)) {
    //     // 打印前几个元素以验证
    //     for (size_t i = 0; i < 10 && i < approximateGT.size(); ++i) {
    //         std::cerr << approximateGT[i] << " ";
    //     }
    //     std::cerr << std::endl;
    // } else {
    //     std::cerr << "读取文件失败" << std::endl;
    // }

    train(X, RandX, ivf, topk, approximateGT, n_leaves, qsize, dataset, soar_lambda);
    
    
    return 0;
}