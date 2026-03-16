#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <cstdio>
#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"
#include <getopt.h>
#include <memory>
#include <pthread.h>
#include <unistd.h>
#include "omp.h"
#include "test_result.h"
#include <dlfcn.h>
#include <cstring>
#include <iomanip>

using namespace std;

constexpr int MAXK = 100;
constexpr int num_threads = 48;//48;
constexpr int nloop = 8;
constexpr bool USE_PTHREAD = true;

long double rotation_time=0;

template<uint32_t D, uint32_t B>
void test(const Matrix<float> &Q, const Matrix<float> &RandQ, const Matrix<float> &X, const Matrix<int64_t> &G, 
            const IVFRN<D, B> &ivf, int k){
    float usr_t, sys_t;
    struct rusage run_start, run_end;

    // ========================================================================
    // Search Parameter
    vector<int> nprobes;
    nprobes.push_back(50);
    // ========================================================================
    
    for(auto nprobe:nprobes){
        float total_time=0;
        float total_ratio=0;
        int correct = 0;

        std::vector<int32_t> labels;
        for(int i=0;i<Q.n;i++){
            GetCurTime( &run_start);
            // std::cerr << "start to search\n";
            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, RandQ.data + i * RandQ.d, k, nprobe);
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
            (void)sys_t; // sys_t is retrieved but not used, suppress warning
            // total_ratio += getRatio(i, Q, X, G, KNNs);
            
            int tmp_correct = 0;
            while(KNNs.empty() == false){
                int id = KNNs.top().second;
                labels.emplace_back(id);
                KNNs.pop();
                for(int j=0;j<k;j++)
                    if(id == G.data[i * G.d + j])tmp_correct ++;
            }
            correct += tmp_correct;
            // std::cerr << "recall = " << tmp_correct << " / " << k << " " << i + 1 << " / " << Q.n << " " << usr_t * 1e6 << "us" << std::endl;
        }
        std::ofstream of2("labels.bin", std::ios::out | std::ios::binary);
        of2.write(reinterpret_cast<char *>(labels.data()), labels.size() * sizeof(int32_t));
        of2.close();
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);
        float average_ratio = total_ratio / (Q.n * k);
        
        cerr << "------------------------------------------------" << endl;
        cerr << "nprobe = " << nprobe << " k = " << k <<  endl;
        cerr << "Recall = " << recall * 100.000 << "%\t" << "Ratio = " << average_ratio << endl;
        cerr << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        
    }
}

/* 全局同步变量 */
pthread_mutex_t mtx;
pthread_cond_t cond;
int ready = 0;

std::vector<int> getAvailableCPUs() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(getpid(), sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        return {};
    }

    std::vector<int> cpuList;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &mask)) {
            cpuList.push_back(i);
        }
    }

    return cpuList;
}

/* 统一线程参数 */
struct UnifiedSearchParams {
    IVFRN<DIM, BB>* ivf;
    float* query;
    float* RandQ;
    int nprobe;
    int topk;
    int* I;
    int n;
    int dim;
    int rand_dim;
    uint32_t pred_nprobe;
    float threshold;
    char* dataset;
    float soar_lambda;
    double thread_time;       /* 线程实际耗时 */
    // float* new_data;
};

void* search_single_thread(void* arg) {
    omp_set_num_threads(1);
    UnifiedSearchParams* params = static_cast<UnifiedSearchParams*>(arg);
    if (params == nullptr) {
        return nullptr;
    }
    pthread_mutex_lock(&mtx);
    while (ready == 0) pthread_cond_wait(&cond, &mtx);
    pthread_mutex_unlock(&mtx);

    if (params->threshold > 0){
    #ifdef __aarch64__
        void* handle = nullptr;
        PredictFunc predFunc = nullptr;
        bool use_adaptive = false;
        if (params->dataset != nullptr) {
            const char* ds = params->dataset;
            if (strstr(ds, "..") == nullptr && strchr(ds, '/') == nullptr && strchr(ds, '\\') == nullptr) {
                char libpath[512];
                std::snprintf(libpath, sizeof(libpath),
                            "data/%s/libadaptivemodel_less.so", params->dataset);
                handle = dlopen(libpath, RTLD_LAZY);
                if (handle != nullptr) {
                    predFunc = reinterpret_cast<PredictFunc>(dlsym(handle, "predict"));
                    if (predFunc != nullptr) {
                        use_adaptive = true;
                    } else {
                        std::cerr << "HDF5 search: dlsym(predict) failed: " << dlerror() << std::endl;
                        dlclose(handle);
                        handle = nullptr;
                    }
                } else {
                    std::cerr << "HDF5 search: dlopen failed: " << dlerror() << std::endl;
                }
            } else {
                std::cerr << "HDF5 search: dataset name must not contain '..', '/', or '\\'" << std::endl;
            }
        }
        if (use_adaptive) {
            for (int i = 0; i < params->n; i++) {
                ResultHeap KNNs = params->ivf->search(
                    params->query + i * params->dim,
                    params->RandQ + i * params->rand_dim,
                    params->topk,
                    params->nprobe,
                    params->soar_lambda,
                    params->threshold,
                    params->pred_nprobe,
                    predFunc);
                int j = 0;
                int32_t* out = params->I + i * params->topk;
                while(KNNs.empty() == false){
                    out[j++] = KNNs.top().second;
                    KNNs.pop();
                }
            }
            if (handle != nullptr) {
                dlclose(handle);
            }
        } else {
            for (int i = 0; i < params->n; i++) {
                ResultHeap KNNs = params->ivf->search(
                    params->query + i * params->dim,
                    params->RandQ + i * params->rand_dim,
                    params->topk,
                    params->nprobe);
                int j = 0;
                int32_t* out = params->I + i * params->topk;
                while(KNNs.empty() == false){
                    out[j++] = KNNs.top().second;
                    KNNs.pop();
                }
            }
        }
    #else
        std::cerr << "Non-Arm devices cannot save query_awareness data." << std::endl << std::endl;
    #endif
    } else {
        for (int i = 0; i < params->n; i++) {
            ResultHeap KNNs = params->ivf->search(
                params->query + i * params->dim,
                params->RandQ + i * params->rand_dim,
                params->topk,
                params->nprobe);
            int j = 0;
            int32_t* out = params->I + i * params->topk;
            while(KNNs.empty() == false){
                out[j++] = KNNs.top().second;
                KNNs.pop();
            }
        }
    }
    return nullptr;
}

int main(int argc, char * argv[]) {
    // check_stack_size();

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
    int nprobe = 0;
    float threshold = 0;
    uint32_t pred_nprobe = 0;
    float soar_lambda = 0;
    
    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:r:k:n:s:p:m:t:e:a:", longopts, &ind);
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
                    nprobe = atoi(optarg);
                    std::cout << "Nprobe set to: " << nprobe << std::endl;
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
            case 't':
                if(optarg){
                    threshold = atof(optarg);
                    std::cout << "threshold set to: " << threshold << std::endl;
                } else {
                    std::cerr << "Warning: -t option requires an argument, using default value 0" << std::endl;
                }
                break; 
            case 'e':
                if(optarg){
                    pred_nprobe = atoi(optarg);
                    std::cout << "pred_nprobe set to: " << pred_nprobe << std::endl;
                } else {
                    std::cerr << "Warning: -e option requires an argument, using default value 0" << std::endl;
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
    

    
    float *xb_;
    float *xq_;
    int64_t *gt_ids_;
    float *gt_dists_;
    int32_t nb_, nq_, dim_, gt_closest;

    loadHDF(data_path, nb_, nq_, dim_, gt_closest, xb_, xq_, gt_ids_, gt_dists_, metric_type);
    delete[] xb_;
    xb_ = nullptr;
    delete[] gt_dists_;
    gt_dists_ = nullptr;

    char transformation_path[256] = "";
    snprintf(transformation_path, sizeof(transformation_path), "%sP_C%d_B%d.fvecs", source, numC, BB);
    Matrix<float> Q(xq_, nq_, dim_, false);
    Matrix<int64_t> G(gt_ids_, nq_, gt_closest, false);
    delete[] xq_;
    xq_ = nullptr;
    delete[] gt_ids_;
    gt_ids_ = nullptr;

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
    // char probe_path[256] = "";
    // sprintf(probe_path, "data/%s/probe_info1.fvecs", dataset);

    // Matrix<float> probe_info(probe_path);
    
    std::cerr << "Loading Succeed!" << std::endl;
    // ================================================================================================================================


    if (freopen(result_file_view, "a", stdout) == nullptr) {
        std::cerr << "Error: failed to open result file: " << result_file_view << std::endl;
        return 1;
    }
    
    IVFRN<DIM, BB> ivf;
#ifdef __aarch64__
    if (soar_lambda == 0){
        // ivf.load(index_path);
        ivf.use_soar = false;
    } else {
        // ivf.load_soar(index_path);
        ivf.use_soar = true;
    }
#endif
    ivf.load(index_path);

    struct rusage run_start, run_end;
    float usr_t, sys_t;
    GetCurTime( &run_start);

    Matrix<float> RandQ(Q.n, BB, Q);

    {
        // Matrix<float> X(xb_, nb_, dim_, false);
        Matrix<float> P(transformation_path);
        RandQ = mul(RandQ, P);
    }

    
    GetCurTime( &run_end);
    GetTime( &run_start, &run_end, &usr_t, &sys_t);
    rotation_time = usr_t * 1e6 / Q.n;
    (void)sys_t; // sys_t is retrieved but not used, suppress warning

    /* 准备结果存储 - 每个线程有自己的结果缓冲区 */
    std::vector<int*> I_vec(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        I_vec[i] = new int[Q.n * topk];
    }


    
    struct timespec start, end;
    TestResult tr;

    std::vector<UnifiedSearchParams> params(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        params[i] = UnifiedSearchParams {
            &ivf,
            Q.data,
            RandQ.data,
            nprobe,
            topk,
            I_vec[i],
            Q.n,
            Q.d,
            RandQ.d,
            pred_nprobe,
            threshold,
            dataset,
            soar_lambda        };
    }
    /* 多轮测试 */
    for (int iter = 0; iter < nloop; iter++) {
        /* 创建线程 */
        if (USE_PTHREAD) {
            std::vector<pthread_t> threads(num_threads);
            std::vector<int> cpu_ids = getAvailableCPUs();
            pthread_mutex_init(&mtx, nullptr);
            pthread_cond_init(&cond, nullptr);
            ready = 0;
            for (int i = 0; i < num_threads; ++i) {
                pthread_create(&threads[i], nullptr, search_single_thread, &params[i]);
                
                /* 线程绑核 */
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cpu_ids[i], &cpuset);
                pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
            }
            /* 同步启动所有线程 */
            sleep(1); /* 确保所有线程已启动 */
            pthread_mutex_lock(&mtx);
            ready = 1;
            pthread_cond_broadcast(&cond);
            pthread_mutex_unlock(&mtx);
            clock_gettime(CLOCK_MONOTONIC, &start);
            for (int i = 0; i < num_threads; i++) {
                pthread_join(threads[i], nullptr);
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
        } else {
            setenv("OMP_PROC_BIND", "true", 1);
            setenv("OMP_PLACES", "cores", 1);
            clock_gettime(CLOCK_MONOTONIC, &start);
            #pragma omp parallel num_threads(num_threads)
            {
                struct timespec startTime, endTime;
                int tid = omp_get_thread_num();
                clock_gettime(CLOCK_MONOTONIC, &startTime);
                search_single_thread(&params[tid]);
                clock_gettime(CLOCK_MONOTONIC, &endTime);
                params[tid].thread_time = (endTime.tv_sec - startTime.tv_sec) + 1e-9 * (endTime.tv_nsec - startTime.tv_nsec);
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
        }

        /* 记录结果 */
        double elapsed_total = (end.tv_sec - start.tv_sec) 
                             + 1e-9 * (end.tv_nsec - start.tv_nsec);
        double avg_thread_time = 0.0;
        double max_thread_time = 0.0;
        double min_thread_time = std::numeric_limits<double>::max();
        
        for (int i = 0; i < num_threads; i++) {
            avg_thread_time += params[i].thread_time;
            if (params[i].thread_time > max_thread_time) 
                max_thread_time = params[i].thread_time;
            if (params[i].thread_time < min_thread_time) 
                min_thread_time = params[i].thread_time;
        }
        avg_thread_time /= num_threads;
        
        // 计算两种QPS
        double qps_total = Q.n / elapsed_total;
        double qps_avg = Q.n / avg_thread_time;

        /* 记录结果 */
        tr.search_time.push_back(elapsed_total);
        tr.total_time += elapsed_total;
        cout << "Loop " << iter + 1 << "/" << nloop << ":\n"
             << "  Total search time = " << elapsed_total << " s\n"
             << "  Thread time stats: min=" << min_thread_time << " s, "
             << "avg=" << avg_thread_time << " s, "
             << "max=" << max_thread_time << " s\n"
             << "  QPS (based on total time) = " << qps_total << "\n"
             << "  QPS (based on thread avg) = " << qps_avg << endl;
        
        /* 清理同步变量 */
        if (USE_PTHREAD) {
            pthread_mutex_destroy(&mtx);
            pthread_cond_destroy(&cond);
        }
    }

    /* 计算召回率 */
    int n_10 = 0;
    for (int iq = 0; iq < Q.n; iq++) {
        for (int i = 0; i < topk; i++) {
            for (int j = 0; j < topk; j++) {
                // std::cerr << "id = " << I_vec[0][iq * topk + j] << std::endl;
                if (I_vec[0][iq * topk + j] == G.data[iq * gt_closest + i]) {
                    n_10++;
                }
            }
        }
    }
    std::cerr << "n_10 = " << n_10 << std::endl;
    tr.recall = n_10 / static_cast<float>(Q.n) / topk;
    std::cerr << "tr.recall = " << tr.recall << std::endl;
    tr.calculate_quantity = Q.n;

    /* 清理资源 */
    for (int i = 0; i < num_threads; ++i) {
        delete[] I_vec[i];
    }
    
    if (nloop >= 4) {
        int begin_id = nloop / 4;       /* 跳过前1/4的循环 */
        int end_id = nloop * 3 / 4;     /* 取中间50%的循环（从1/4到3/4） */
        tr.reorder();                   /* 对搜索时间进行排序并计算累积时间 */
        double qps = tr.calculate_quantity / tr.get_total(begin_id, end_id) * (end_id - begin_id);
        cerr << "Final Results: qps " << std::fixed << std::setprecision(1) << qps << " recall " << std::setprecision(5) << tr.recall << endl;
    } else {
        double qps = tr.calculate_quantity * nloop / tr.total_time;
        cerr << "Final Results: qps " << std::fixed << std::setprecision(0) << qps << " recall " << std::setprecision(5) << tr.recall << endl;
    }

    return 0;
}