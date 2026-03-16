#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <iostream>
#include <cstdio>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"

using namespace std;

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'd'},
        {"source",                      required_argument, 0, 's'},

        {"data_path",                   required_argument, 0, 'p'},
        {"metric_type",                      required_argument, 0, 'm'},
        {"soar_lambda",                      required_argument, 0, 'a'},
};

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)
    float soar_lambda = 0;

    char dataset[256]="";
    char source[256]="";
    char data_path[256] = ""; 
    char metric_type[256] = "";
    char scan_type[256] = "";

#if defined(FAST_SCAN)
    strcpy(scan_type, "fastscan");
#elif defined(SCAN)
    strcpy(scan_type, "scan");
#endif    
    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:s:p:m:a:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg){
                    strcpy(dataset, optarg);
                }
                break;
            case 's':
                if(optarg){
                    strcpy(source, optarg);
                }
                break;
            case 'p':
                if(optarg){
                    strcpy(data_path, optarg);
                }
                break;
            case 'm':
                if(optarg){
                    strcpy(metric_type, optarg);
                }
                break;           
            case 'a':
                if(optarg){
                    soar_lambda = atof(optarg);
                    std::cout << "soar_lambda set to: " << soar_lambda << std::endl;
                }
                break; 
        }
    }

    if (soar_lambda != 0) {
        std::cerr << "SOAR indexing is not supported in this branch" << std::endl;
        return 1;
    }

    
    // ==============================================================================================================
    // Load Data
    char index_path[256] = "";
    char centroid_path[256] = "";
    char x0_path[256] = "";
    char dist_to_centroid_path[256] = "";
    char cluster_id_path[256] = "";
    char binary_path[256] = "";

    float *xb_;
    int32_t nb_, dim_;

    loadHDFBase(data_path, nb_, dim_, xb_, metric_type);

    Matrix<float> X(xb_, nb_, dim_, false);
    delete[] xb_;
    xb_ = nullptr;

    std::cerr << "BB= " <<BB<< std::endl;
    snprintf(centroid_path, sizeof(centroid_path), "%sRandCentroid_C%d_B%d.fvecs", source, numC, BB);
    Matrix<float> C(centroid_path);

    snprintf(x0_path, sizeof(x0_path), "%sx0_C%d_B%d.fvecs", source, numC, BB);
    Matrix<float> x0(x0_path);

    snprintf(dist_to_centroid_path, sizeof(dist_to_centroid_path), "%s%s_dist_to_centroid_%d.fvecs", source, dataset, numC);
    Matrix<float> dist_to_centroid(dist_to_centroid_path);
    
    snprintf(cluster_id_path, sizeof(cluster_id_path), "%s%s_cluster_id_%d.ivecs", source, dataset, numC);
    Matrix<uint32_t> cluster_id(cluster_id_path);
    
    snprintf(binary_path, sizeof(binary_path), "%sRandNet_C%d_B%d.Ivecs", source, numC, BB);
    Matrix<uint64_t> binary(binary_path);

    snprintf(index_path, sizeof(index_path), "%sivfrabitq_%s_%d_B%d.index", source, scan_type, numC, BB);
    std::cerr << "Loading Succeed!" << std::endl << std::endl;
    // ==============================================================================================================

    IVFRN<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
    std::cerr << "start to save\n";
    ivf.save(index_path);
    
    return 0;
}