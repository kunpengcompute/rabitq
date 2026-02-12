#!/bin/bash

# 参数检查
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <dataset> [fastscan|scan] [generate|index|search|train|eval|all]"
  echo "Datasets: sift | deep | glove | fashion | gist"
  echo "Modes: fastscan (default) | scan"
  echo "Stages: generate | index | search | train | eval | all (default: all stages)"
  exit 1
fi

DATASET_NAME="$1"
MODE="${2:-fastscan}"
STAGE="${3:-all}"

ARCH=$(uname -m)

# 当前路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 数据集路径
BASE_DATASET_PATH="${SCRIPT_DIR}/datasets"

# 数据集配置
case "${DATASET_NAME}" in
  sift)
    HDF5_FILENAME="sift-128-euclidean.hdf5"
    METRIC_TYPE="squared_l2"
    D=128
    B=128
    ;;
  deep)
    HDF5_FILENAME="deep-image-96-angular.hdf5"
    METRIC_TYPE="dot_product"
    D=96
    B=128
    ;;
  glove)
    HDF5_FILENAME="glove-100-angular.hdf5"
    METRIC_TYPE="dot_product"
    D=100
    B=128
    ;;
  fashion)
    HDF5_FILENAME="fashion-mnist-784-euclidean.hdf5"
    METRIC_TYPE="squared_l2"
    D=784
    B=832
    ;;
  gist)
    HDF5_FILENAME="gist-960-euclidean.hdf5"
    METRIC_TYPE="squared_l2"
    D=960
    B=960
    ;;
  *)
    echo "Invalid dataset: ${DATASET_NAME}. Available: sift | deep | glove | fashion | gist"
    exit 2
    ;;
esac

HDF5_PATH="${BASE_DATASET_PATH}/${HDF5_FILENAME}"

# 模式配置
EXTRA_DEFS=""
case "${MODE}" in
  fastscan) EXTRA_DEFS="-D FAST_SCAN" ;;
  scan)     EXTRA_DEFS="-D SCAN" ;;
  *) echo "Invalid mode: ${MODE}. Use FastScan or Scan"; exit 3 ;;
esac
  if [[ "$ARCH" == "x86_64" ]]; then
    declare -A DATASET_PARAMS=(
        ["sift"]="2048 77 0 0 0"
        ["deep"]="4096 95 0 0 0"
        ["glove"]="2048 800 0 0 0"
        ["fashion"]="128 6 0 0 0" 
        ["gist"]="2048 210 0 0 0"
    )
  elif [[ "$ARCH" == "aarch64" ]]; then
    declare -A DATASET_PARAMS=(
        ["sift"]="2048 75 0.4 40 0"
        ["deep"]="4096 34 0 0 1.2"
        ["glove"]="2048 765 0 0 0"
        ["fashion"]="128 6 0 0 0"
        ["gist"]="2048 100 0.34 43 1.2"
    )
  fi

read K_VALUE NPROBE THRESHOLD PRED_NPROBE SOAR_LAMBDA <<< "${DATASET_PARAMS[$DATASET_NAME]}"

source='./data'

echo "=== 配置信息 ==="
echo "模式: ${MODE}"
echo "阶段: ${STAGE}"
echo "NPROBE: ${NPROBE}"
echo "THRESHOLD: ${THRESHOLD}"
echo "PRED_NPROBE: ${PRED_NPROBE}"
echo "HDF5_PATH: ${HDF5_PATH}"
echo "数据集: ${DATASET_NAME}"
echo "K_VALUE: ${K_VALUE}"
echo "METRIC_TYPE: ${METRIC_TYPE}"
echo "SOAR_LAMBDA: ${SOAR_LAMBDA}"
echo "================="

# 生成数据
generate() {
    echo "=== 开始生成数据 ==="
    cd ./data || exit 1
    # mkdir -p sift glove gist deep fashion
    numactl -N 0 -m 0 python3 ivf.py $HDF5_PATH $DATASET_NAME $K_VALUE $METRIC_TYPE $SOAR_LAMBDA
    numactl -N 0 -m 0 python3 rabitq.py $HDF5_PATH $DATASET_NAME $K_VALUE $METRIC_TYPE $SOAR_LAMBDA
    
    cd ..
    echo "=== 数据生成完成 ==="
}

# 构建索引
index() {
    echo "=== 开始构建索引 ==="
    mkdir -p bin
    if [[ "$ARCH" == "x86_64" ]]; then
      g++ -o ./bin/index_${DATASET_NAME} ./src/index.cpp \
          -I ./src/ \
          -I /usr/include/hdf5/serial/ \
          -L /usr/lib/aarch64-linux-gnu/hdf5/serial/ \
          -lhdf5_cpp -lhdf5 \
          -O3 -march=native \
          -D BB=${B} -D DIM=${D} -D numC=${K_VALUE} -D B_QUERY=4 ${EXTRA_DEFS}
    elif [[ "$ARCH" == "aarch64" ]]; then
      g++ -o ./bin/index_${DATASET_NAME} ./src/index.cpp \
          -I ./src/ \
          -I /usr/include/hdf5/serial/ \
          -L /usr/lib/aarch64-linux-gnu/hdf5/serial/ \
          -lhdf5_cpp -lhdf5 \
          -O3 -march=armv8.2-a+fp16fml+dotprod -fpermissive \
          -D BB=${B} -D DIM=${D} -D numC=${K_VALUE} -D B_QUERY=4 ${EXTRA_DEFS}
    fi
    
    numactl -N 0 -m 0 ./bin/index_${DATASET_NAME} -d $DATASET_NAME -s "$source/$DATASET_NAME/" -p "$HDF5_PATH" -m "$METRIC_TYPE" -a "$SOAR_LAMBDA"
    
    echo "=== 索引构建完成 ==="
}

# 执行搜索
search() {
    echo "=== 开始执行搜索 ==="
    local k=10
    mkdir -p bin
    if [[ "$ARCH" == "x86_64" ]]; then
      clang++ -g -march=native  -fpermissive -ffast-math -fopenmp -Ofast \
              -o ./bin/search_${DATASET_NAME} \
              ./src/search.cpp ./src/test_result.cpp \
              -I ./src/ \
              -I /usr/include/hdf5/serial/ \
              -L /usr/lib/aarch64-linux-gnu/hdf5/serial/ \
              -lhdf5_cpp -lhdf5 \
              -D BB=${B} -D DIM=${D} -D numC=${K_VALUE} -D B_QUERY=4 ${EXTRA_DEFS}
    elif [[ "$ARCH" == "aarch64" ]]; then
      clang++ -g -march=armv8-a+fp16fml -falign-loops=64  -fpermissive -ffast-math -fno-trapping-math -funroll-loops -fopenmp -Ofast -flto=full -fuse-ld=lld -Wno-c++11-narrowing \
              -o ./bin/search_${DATASET_NAME} \
              ./src/search.cpp ./src/test_result.cpp ./src/krl_table_lookup_fast_scan.s \
              -I ./src/ \
              -I /usr/include/hdf5/serial/ \
              -L /usr/lib/aarch64-linux-gnu/hdf5/serial/ \
              -lhdf5_cpp -lhdf5 \
              -I`jemalloc-config --includedir` \
              -L`jemalloc-config --libdir` -Wl,-rpath,`jemalloc-config --libdir` \
              -ljemalloc `jemalloc-config --libs` \
              -D BB=${B} -D DIM=${D} -D numC=${K_VALUE} -D B_QUERY=4 ${EXTRA_DEFS}
    fi
    result_path=./results
    mkdir -p ${result_path}
    res="${result_path}/${DATASET_NAME}/"
    mkdir -p "$result_path/${DATASET_NAME}/"

    export MALLOC_CONF="narenas:1"
    
    numactl -N 0 -m 0 ./bin/search_${DATASET_NAME} \
        -d ${DATASET_NAME} -r ${res} -k ${k} -n ${NPROBE} \
        -s "$source/$DATASET_NAME/" -p "$HDF5_PATH" -m "$METRIC_TYPE" -t "$THRESHOLD" -e "$PRED_NPROBE" -a "$SOAR_LAMBDA" \
         &
   
    numactl -N 1 -m 1 ./bin/search_${DATASET_NAME} \
        -d ${DATASET_NAME} -r ${res} -k ${k} -n ${NPROBE} \
        -s "$source/$DATASET_NAME/" -p "$HDF5_PATH" -m "$METRIC_TYPE" -t "$THRESHOLD" -e "$PRED_NPROBE" -a "$SOAR_LAMBDA" \
        &
    
    numactl -N 2 -m 2 ./bin/search_${DATASET_NAME} \
        -d ${DATASET_NAME} -r ${res} -k ${k} -n ${NPROBE} \
        -s "$source/$DATASET_NAME/" -p "$HDF5_PATH" -m "$METRIC_TYPE" -t "$THRESHOLD" -e "$PRED_NPROBE" -a "$SOAR_LAMBDA" \
        &
    
    numactl -N 3 -m 3 ./bin/search_${DATASET_NAME} \
        -d ${DATASET_NAME} -r ${res} -k ${k} -n ${NPROBE} \
        -s "$source/$DATASET_NAME/" -p "$HDF5_PATH" -m "$METRIC_TYPE" -t "$THRESHOLD" -e "$PRED_NPROBE" -a "$SOAR_LAMBDA" \
        &
    
    wait
    echo "=== 搜索完成 ==="
}


# 训练模型准备数据
train() {
    echo "=== 开始执行训练 ==="
    local k=10
    mkdir -p bin
    if [[ "$ARCH" == "aarch64" ]]; then
      clang++ -g -march=armv8-a+fp16fml -fpermissive -ffast-math -fno-trapping-math -funroll-loops -fopenmp -Ofast -flto=full -fuse-ld=lld -Wno-c++11-narrowing\
              -o ./bin/search_model_${DATASET_NAME} \
              ./src/search_model.cpp ./src/test_result.cpp ./src/krl_table_lookup_fast_scan.s \
              -I ./src/ \
              -I /usr/include/hdf5/serial/ \
              -L /usr/lib/aarch64-linux-gnu/hdf5/serial/ \
              -lhdf5_cpp -lhdf5 \
              -I`jemalloc-config --includedir` \
              -L`jemalloc-config --libdir` -Wl,-rpath,`jemalloc-config --libdir` \
              -ljemalloc `jemalloc-config --libs` \
              -D BB=${B} -D DIM=${D} -D numC=${K_VALUE} -D B_QUERY=4 ${EXTRA_DEFS}

      result_path=./results
      mkdir -p ${result_path}
      res="${result_path}/${DATASET_NAME}/"
      mkdir -p "$result_path/${DATASET_NAME}/"
      
      numactl -N 0 -m 0  ./bin/search_model_${DATASET_NAME} \
          -d ${DATASET_NAME} -r ${res} -k ${k} -n ${K_VALUE} \
          -s "$source/$DATASET_NAME/" -p "$HDF5_PATH" -m "$METRIC_TYPE" -a "$SOAR_LAMBDA"
    fi
    echo "=== 搜索完成 ==="
}

# train model
eval() {
    echo "=== 开始生成数据 ==="
    if [[ "$ARCH" == "aarch64" ]]; then
      cd ./data || exit 1
      LD_PRELOAD="/root/miniconda3/lib/libtreelite.so" numactl -N 0 -m 0 python3 eval.py "$HDF5_PATH" "$DATASET_NAME" "$K_VALUE" "$METRIC_TYPE" "$source/$DATASET_NAME/" "${B}"
      cd ..
    fi
    echo "=== 数据生成完成 ==="
}
# 执行流程控制
case "${STAGE}" in
  generate)
    generate
    ;;
  index)
    index
    ;;
  search)
    search
    ;;
  train)
    train
    ;;
  eval)
    eval
    ;;
  all)
    generate
    index
    train
    eval
    search
    ;;
  *)
    echo "无效的阶段: ${STAGE}"
    echo "可用选项: generate | index | search | train | eval | all"
    exit 4
    ;;
esac
