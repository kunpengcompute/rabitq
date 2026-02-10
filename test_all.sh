#!/usr/bin/env bash

# 定义数据集列表
#datasets=("fashion" "glove" "sift" "gist" "deep")
datasets=("fashion" "sift" "gist" "deep" "glove")

# 定义操作列表
#operations=("generate" "index" "search")
# operations=("search")
if [ $# -eq 0 ]; then
  echo "Usage: $0 <operation1> [operation2] ..."
  echo "Available operations: generate | index | search | train | eval | all"
  exit 1
fi
operations=("$@")

# 遍历每个数据集和操作
for dataset in "${datasets[@]}"; do
  for operation in "${operations[@]}"; do
    echo "Running $operation on $dataset dataset..."
    bash run.sh $dataset fastscan $operation
    if [ $? -ne 0 ]; then
      echo "Error occurred while running $operation on $dataset dataset."
      exit 1
    fi
  done
done

echo "All operations completed successfully."
