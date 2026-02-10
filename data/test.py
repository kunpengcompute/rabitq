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
import h5py

from utils.io import *

def normalize_vector(data: MutableSequence[float]) -> None:
    size = len(data)
    if size == 0:
        return

    norm = math.sqrt(sum(x * x for x in data))

    if norm == 0.0:
        scale = 1.0 / math.sqrt(size)
        for i in range(size):
            data[i] = scale
    else:
        for i in range(size):
            data[i] /= norm


X = np.random.rand(100, 128).astype(np.float32)
for row in X:          # row 是 view，原地修改
    normalize_vector(row)

print(X)


