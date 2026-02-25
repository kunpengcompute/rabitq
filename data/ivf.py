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
import faiss
import struct
import os
import h5py
from utils.io import *
import math
from typing import Sequence, MutableSequence
import sys

source = './'

def assign_spilled_vec_fast(X, C, labels, lam=0.5, batch=8_000):
    N, K1, D = X.shape[0], C.shape[0], X.shape[1]
    spilled  = np.empty(N, dtype=np.int32)
    spilled_loss = np.empty(N, dtype=np.float32)

    r        = X - C[labels]                       # (N, D)
    r_norm2  = np.einsum('nd,nd->n', r, r)         # (N,)

    for i in range(0, N, batch):
        s, e = i, min(i+batch, N)
        xb = X[s:e]                                # (B, D)
        rb = r[s:e]                                # (B, D)
        lb = labels[s:e]                           # (B,)
        B  = xb.shape[0]

        # 1. ||r'||² = ||x - c'||²
        #    = ||x||² - 2<x,c'> + ||c'||²
        x_norm2  = np.einsum('bd,bd->b', xb, xb)[:, None]     # (B,1)
        c_norm2  = np.einsum('kd,kd->k', C, C)[None, :]        # (1,K1)
        xdotc    = xb @ C.T                                     # (B,K1)
        term1    = x_norm2 - 2 * xdotc + c_norm2               # (B,K1)

        # 2. rb·r' = rb·(x - c') = rb·x - rb·c'
        rb_dot_x  = np.einsum('bd,bd->b', rb, xb)[:, None]      # (B,1)
        rb_dot_c  = rb @ C.T                                    # (B,K1)
        dot_r_rp  = rb_dot_x - rb_dot_c                         # (B,K1)

        # 3. 投影能量 & 损失
        proj_num = dot_r_rp ** 2
        proj_den = r_norm2[s:e, None] + 1e-16
        term2    = proj_num / proj_den
        loss     = term1 + lam * term2
        loss[np.arange(B), lb] = np.inf

        min_idx                  = np.argmin(loss, axis=1)
        spilled[s:e]             = min_idx
        spilled_loss[s:e]        = loss[np.arange(B), min_idx]
    return spilled, spilled_loss

def test_ip(hdf5_path, dataset_name, K_value, metric_type, soar_lambda):
    dataset = dataset_name
    with h5py.File(hdf5_path, "r") as fr:
        train = np.array(fr["train"]).astype(np.float32)
    
    print("train = ", train)
    path = os.path.join(source, dataset)
    X = train
    if metric_type == "dot_product":
        norms = np.linalg.norm(X, axis=1, keepdims=True)          # (100,1)
        np.divide(X, norms, out=X, where=norms!=0)                # 原地归一化

        # For zero-norm vectors, set to uniform vector with norm 1
        # Use actual dimension instead of hardcoded 128
        X[norms.squeeze()==0] = 1.0 / np.sqrt(X.shape[1])

    print("X = ", X)
    D = X.shape[1]
    K = K_value
    print("K = ", K)
    centroids_path = os.path.join(path, f'{dataset}_centroid_{K}.fvecs')
    dist_to_centroid_path = os.path.join(path, f'{dataset}_dist_to_centroid_{K}.fvecs')
    cluster_id_path = os.path.join(path, f'{dataset}_cluster_id_{K}.ivecs')

    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    
    index.cp.niter = 100
    index.cp.verbose = True
    
    index.train(X)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
    dist_to_centroid = dist_to_centroid ** 0.5

    to_fvecs(dist_to_centroid_path, dist_to_centroid)
    to_ivecs(cluster_id_path, cluster_id)
    to_fvecs(centroids_path, centroids)

    # centroids  = read_fvecs(centroids_path)
    # print("centroids = ", centroids)
    # cluster_id = read_ivecs(cluster_id_path)
    # print("cluster_id = ", cluster_id)

    if (soar_lambda > 0):
        print("soar_lambda = ", soar_lambda)
        #### SOAR
        # soar_lambda = 1.2
        spilled_labels, spilled_loss = assign_spilled_vec_fast(X, centroids, cluster_id.ravel(), soar_lambda)

        dist_to_spilled_labels = []
        for i, x in enumerate(spilled_labels):
            dist = (centroids[x] - X[i]) * (centroids[x] - X[i])
            dist_to_spilled_labels.append(np.sum(dist) ** 0.5)

        spilled_labels = np.array(spilled_labels).reshape(-1,1)
        dist_to_spilled_labels = np.array(dist_to_spilled_labels).reshape(-1,1)

        spilled_labels_path = os.path.join(path, f'{dataset}_spilled_labels_{K}.ivecs')
        dist_to_spilled_labels_path = os.path.join(path, f'{dataset}_dist_to_spilled_labels_{K}.fvecs')

        to_ivecs(spilled_labels_path, spilled_labels)
        to_fvecs(dist_to_spilled_labels_path, dist_to_spilled_labels)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python ivf.py <hdf5_path> <dataset_name> <K_value> <METRIC_TYPE> <SOAR_LAMBDA>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    dataset_name = sys.argv[2]
    K_value = int(sys.argv[3])
    metric_type = sys.argv[4]
    soar_lambda = float(sys.argv[5])

    
    test_ip(hdf5_path, dataset_name, K_value, metric_type, soar_lambda)

