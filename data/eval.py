"""
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
"""
import numpy as np
import os
from utils.io import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import h5py

import sys
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import treelite
import tl2cgen

import time
from sklearn.model_selection import GridSearchCV
import treelite
import tl2cgen


# 提取特征
def extract_features(dist_to_centroid,data_ori, p1_leaf):
    mean = np.mean(data_ori, axis=1)
    std_val = np.std(data_ori, axis=1)
    sparsity = np.count_nonzero(data_ori, axis=1) / D
    norms = np.linalg.norm(data_ori, axis=1)
    return np.column_stack((mean, std_val, sparsity, norms, dist_to_centroid, p1_leaf))

def extract_features_base(data_ori):
    mean = np.mean(data_ori, axis=1)
    std_val = np.std(data_ori, axis=1)
    sparsity = np.count_nonzero(data_ori, axis=1) / D
    norms = np.linalg.norm(data_ori, axis=1)
    return np.column_stack((mean, std_val, sparsity, norms))

# 计算 L2 距离
def compute_l2_distances(base, centroids):
    distances = np.linalg.norm(base[:, np.newaxis] - centroids, axis=2)
    return distances

def compute_l2_distances_chunked(base, centroids, chunk_rows=1024):
    n, d = base.shape
    k = centroids.shape[0]
    base_norm2  = np.einsum('ij,ij->i', base, base, dtype=np.float32)      # (n,)
    cent_norm2  = np.einsum('ij,ij->i', centroids, centroids, dtype=np.float32)  # (k,)
    
    dist = np.empty((n, k), dtype=np.float32)
    for i in range(0, n, chunk_rows):
        end = min(i + chunk_rows, n)
        dist[i:end] = (base_norm2[i:end, np.newaxis] +
                       cent_norm2[np.newaxis, :] -
                       2 * np.matmul(base[i:end], centroids.T))
    dist = np.sqrt(dist)
    return dist

def compute_ip_distances(base, centroids, chunk_rows=1024):
    n, d = base.shape
    k = centroids.shape[0]
    ip = np.empty((n, k), dtype=np.float32)
    for i in range(0, n, chunk_rows):
        end = min(i + chunk_rows, n)
        ip[i:end] = base[i:end] @ centroids.T
    return ip

# 获取最近的 8 个质心及其距离
def get_nearest_centroids(distances, k=8):
    # 对每个向量的 L2 距离进行排序，获取最近的 k 个质心及其距离
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.sort(distances, axis=1)[:, :k]
    return nearest_indices, nearest_distances

def processProbeData(probe, nearest_indices, nlist, i, using_probe):
    leaf_multplier = 1
    for j in range(using_probe):
        if (nearest_indices[i][j*leaf_multplier] >= nlist):
            print("AdaptiveModel: out of n_leaves range: ", nearest_indices[i][j*leaf_multplier])
            continue
        probe[nearest_indices[i][j*leaf_multplier]] += 1

def prepareProbeInfo(base_label, nearest_indices, nlist, train50p):
    probe_info = [1] * nlist
    probe_normal = [1] * nlist
    print("base_label.shape[0] = ", base_label.shape[0])
    for i in range(base_label.shape[0]):
        if base_label[i] > train50p:
            processProbeData(probe_info, nearest_indices, nlist, i, 8)
        else:
            processProbeData(probe_normal, nearest_indices, nlist, i, 8)
    for i in range(nlist):
        probe_info[i] = probe_normal[i] / (probe_normal[i] + probe_info[i])
    probe_normal.clear()
    
    return probe_info

def translate(nearest_indices, probe_info):
    
    p1_leaf = np.zeros((nearest_indices.shape[0], 8), dtype=np.float32)
    print("p1_leaf.shape = ", p1_leaf.shape)
    for i in range(nearest_indices.shape[0]):
        for j in range(8):

            p1_leaf[i][j] = probe_info[nearest_indices[i][j]]
    return p1_leaf

def eval_model(gbm):
    threshold = 0.5
    y_pred = gbm.predict(X_test)
    print("[117] threshold = ", threshold)
    print("------------------------> y_pred_25 = ", np.percentile(y_pred, 25))
    print("------------------------> y_pred_50 = ", np.percentile(y_pred, 50))
    print("------------------------> y_pred_75 = ", np.percentile(y_pred, 75))

    y_pred_label = np.where(y_pred > threshold, 1, 0)
    # Accuracy
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_label))
    confusion_matrix_result = metrics.confusion_matrix(y_test, y_pred_label)

    # Confusion matrix
    print('Confusion matrix result:\n', confusion_matrix_result)

    # AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print(f'roc_auc: {roc_auc}')

    # best threshold
    y_threshold = tpr - fpr
    Youden_index = np.argmax(y_threshold)  # Only the first occurrence is returned.
    optimal_threshold = thresholds[Youden_index]
    optimal_point = [fpr[Youden_index], tpr[Youden_index]]
    print(f'optimal Threshold:{optimal_threshold:.2f}  fpr:{fpr[Youden_index]} tpr:{tpr[Youden_index]}')

    return roc_auc

def mode_factory(params):
    evals_result = {}
    cur_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'is_unbalance': 'true',
        'early_stopping_rounds': 20,
        'verbose': -1,
        **params
    }

    gbm = lgb.train(cur_params,
                    train_data,
                    num_boost_round=200,
                    valid_sets=[validation_data])
    return gbm




if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage: python eval.py <hdf5_path> <dataset_name> <K_value> <METRIC_TYPE> <data_path> <BB>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    dataset_name = sys.argv[2]
    K_value = int(sys.argv[3])
    metric_type = sys.argv[4]
    all_data_path = sys.argv[5]
    BB = int(sys.argv[6])

    print("K_value = ", K_value)
    print("all_data_type = ", all_data_path)
    print("BB = ", BB)

    projection = read_fvecs(dataset_name+'/P_C' + str(K_value) + '_B' + str(BB) + '.fvecs')
    print("projection = ", projection.shape)

    centroids = read_fvecs(dataset_name + '/RandCentroid_C'+str(K_value) + '_B' + str(BB)+'.fvecs')
    print("centroids = ", centroids.shape)

    with h5py.File(hdf5_path, "r") as fr:
        base = np.array(fr["train"]).astype(np.float32)

    D = base.shape[1]  
    print("metric_type = ",  metric_type)
    if metric_type == "dot_product":
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        np.divide(base, norms, out=base, where=norms!=0)             

        # For zero-norm vectors, set to uniform vector with norm 1
        # Use actual dimension instead of hardcoded 128
        base[norms.squeeze()==0] = 1.0 / np.sqrt(D)

    base_ori = base[:50000]
    if (BB > D):
        base = np.pad(base, ((0, 0), (0, BB - D)), mode='constant', constant_values=0)
    base_proj = np.dot(base, projection)[:50000]
    
    print("base_proj = ", base_proj.shape)
    K = K_value

    distances = compute_l2_distances_chunked(base_proj, centroids)
    print("distances shape = ", distances.shape)

    base_label = np.fromfile(dataset_name + "/approximateGT.bin",dtype=np.int64).reshape(-1,10)

    expectedNprobe = np.fromfile(dataset_name + "/expectedNprobe1.bin",dtype=np.int64)

    print("expectedNprobe max = ", np.max(expectedNprobe))
    print("expectedNprobe min = ", np.min(expectedNprobe))
    
    train_features = extract_features_base(base_ori)

    print("train_features shape = ", train_features.shape)

    # Use 50th percentile (median) as threshold for binary classification
    threshold = np.percentile(expectedNprobe, 50)
    print("threshold (50th percentile) = ", threshold)
    print("expectedNprobe = ", expectedNprobe)

    y = np.where(expectedNprobe > threshold, 1, 0)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(train_features, y, test_size=0.2, random_state=2020)

    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, label=y_test)

    base_params = {
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.06,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
    }

    model = mode_factory(base_params)
    auc_base = eval_model(model)

    model = treelite.frontend.from_lightgbm(model)

    tl2cgen.export_lib(model, toolchain='gcc', libpath= dataset_name + "/libadaptivemodel_less.so", params={'parallel_comp':32})
