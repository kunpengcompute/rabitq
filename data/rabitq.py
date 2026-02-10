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
import struct
import time
import os
from utils.io import *
from tqdm import tqdm
import h5py
import sys

source = './'

def Orthogonal(D):
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

def GenerateBinaryCode(X, P):
    XP = np.dot(X, P)
    binary_XP = (XP > 0)
    X0 = np.sum(XP * (2 * binary_XP - 1) / D ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1, keepdims=True)
    return binary_XP, X0


def test_ip(hdf5_path, dataset_name, K_value, metric_type, soar_lambda):
    
    with h5py.File(hdf5_path, "r") as fr:
        train = np.array(fr["train"]).astype(np.float32)

    # X = train
    X = train
    if metric_type == "dot_product":
        norms = np.linalg.norm(X, axis=1, keepdims=True)          # (100,1)
        np.divide(X, norms, out=X, where=norms!=0)                # 原地归一化

        # For zero-norm vectors, set to uniform vector with norm 1
        # Use actual dimension instead of hardcoded 128
        X[norms.squeeze()==0] = 1.0 / np.sqrt(X.shape[1])

    dataset = dataset_name
    print("dataset = ", dataset)
    path                    = os.path.join(source, dataset)
    C = K_value
    print("C = ", C)
    centroids_path          = os.path.join(path, f'{dataset}_centroid_{C}.fvecs')
    dist_to_centroid_path   = os.path.join(path, f'{dataset}_dist_to_centroid_{C}.fvecs')
    cluster_id_path         = os.path.join(path, f'{dataset}_cluster_id_{C}.ivecs')

    centroids  = read_fvecs(centroids_path)
    cluster_id = read_ivecs(cluster_id_path)
    
    D = X.shape[1]
    B = (D + 63) // 64 * 64
    MAX_BD = max(D, B)
    print("D = ", D)

    print("B = ", B)

    projection_path          = os.path.join(path, f'P_C{C}_B{B}.fvecs')
    randomized_centroid_path = os.path.join(path, f'RandCentroid_C{C}_B{B}.fvecs')
    RN_path                  = os.path.join(path, f'RandNet_C{C}_B{B}.Ivecs')
    x0_path                  = os.path.join(path, f'x0_C{C}_B{B}.fvecs')

    X_pad         = np.pad(X, ((0, 0), (0, MAX_BD-D)), 'constant')
    centroids_pad = np.pad(centroids, ((0, 0), (0, MAX_BD-D)), 'constant')
    np.random.seed(0)

    # The inverse of an orthogonal matrix equals to its transpose. 
    P = Orthogonal(MAX_BD)
    P = P.T
    
    cluster_id=np.squeeze(cluster_id)
    XP = np.dot(X_pad, P)
    CP = np.dot(centroids_pad, P)
    XP = XP - CP[cluster_id]
    bin_XP = (XP > 0)
    
    # The inner product between the data vector and the quantized data vector, i.e., <\bar o, o>.
    x0 = np.sum(XP[ : , :B] * (2 * bin_XP[ : , :B] - 1) / B ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP, axis=1, keepdims=True)
    
    # To remove illy defined x0
    # np.linalg.norm(XP, axis=1, keepdims=True) = 0 indicates that its estimated distance based on our method has no error.
    # Thus, it should be good to set x0 as any finite non-zero number.  
    x0[~np.isfinite(x0)] = 0.8
    
    bin_XP = bin_XP[:, :B].flatten()
    uint64_XP = np.packbits(bin_XP.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
    uint64_XP = uint64_XP.reshape(-1, B >> 6)

    # Output
    to_fvecs(randomized_centroid_path, CP)
    to_Ivecs(RN_path                 , uint64_XP)
    to_fvecs(x0_path                 , x0)
    to_fvecs(projection_path         , P)
    
    if (soar_lambda > 0) :
        #### SOAR
        spilled_labels_path = os.path.join(path, f'{dataset}_spilled_labels_{C}.ivecs')
        dist_to_spilled_labels_path = os.path.join(path, f'{dataset}_dist_to_spilled_labels_{C}.fvecs')

        spilled_id = read_ivecs(spilled_labels_path)
        dist_to_spilled_label = read_fvecs(dist_to_spilled_labels_path)
        print("cluster_id shape = ", cluster_id.shape)
        print("spilled_id shape = ", spilled_id.shape)

        spilled_id=np.squeeze(spilled_id)
        XP_spilled = np.dot(X_pad, P)
        XP_spilled = XP_spilled - CP[spilled_id]
        bin_XP_spilled = (XP_spilled > 0)

        x0_spilled = np.sum(XP_spilled[ : , :B] * (2 * bin_XP_spilled[ : , :B] - 1) / B ** 0.5, axis=1, keepdims=True) / np.linalg.norm(XP_spilled, axis=1, keepdims=True)

        bin_XP_spilled = bin_XP_spilled[:, :B].flatten()
        uint64_XP_spilled = np.packbits(bin_XP_spilled.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
        uint64_XP_spilled = uint64_XP_spilled.reshape(-1, B >> 6)

        RN_spilled_path                  = os.path.join(path, f'RandNet_spilled_C{C}_B{B}.Ivecs')
        x0_spilled_path                  = os.path.join(path, f'x0_spilled_C{C}_B{B}.fvecs')
        to_Ivecs(RN_spilled_path                 , uint64_XP_spilled)
        to_fvecs(x0_spilled_path                 , x0_spilled)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python ivf.py <hdf5_path> <dataset_name> <K_value> <METRIC_TYPE> <SOAR_LAMBDA>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    dataset_name = sys.argv[2]
    K_value = int(sys.argv[3])
    metric_type = sys.argv[4]
    soar_lambda = float(sys.argv[5])
    
    test_ip(hdf5_path, dataset_name, K_value, metric_type, soar_lambda)

