# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
# This code was originally written by FNNDSC/BCH and is part of the fetal_brain_assessment
# repository at ttps://github.com/FNNDSC/pl-fetal-brain-assessment/blob/main/fetal_brain_assessment/predict_resnet.py
# (originally licensed under an MIT license)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Preprocessing for the image quality assessment based  on FNNDSC's implementation.
"""
import numpy as np


def center_crop_to_shape(data, target):
    """Center cropping of data to a target shape.
    This is required for the input of fnndsc's stack-wise DL IQA.
    """
    x, y, z = data.shape
    tx, ty, tz = target
    if x > tx:
        startx = x // 2 - tx // 2
        data = data[startx : startx + tx]
    if y > ty:
        starty = y // 2 - ty // 2
        data = data[:, starty : starty + ty]
    if z > tz:
        startz = z // 2 - tz // 2
        data = data[:, :, startz : startz + tz]
    assert all(x <= t for x, t in zip(data.shape, target))
    return data


def fnndsc_preprocess(data, mask):
    """ """
    data = data * mask
    # Detect the bounding box of the foreground
    idx = np.nonzero(data > 0)
    x1, x2 = idx[0].min(), idx[0].max()
    y1, y2 = idx[1].min(), idx[1].max()
    z1, z2 = idx[2].min(), idx[2].max()
    data = data[x1:x2, y1:y2, z1:z2]
    if any(x > t for x, t in zip(data.shape, (217, 178, 60))):
        print("\tWARNING: Data exceed dimensions (217, 178, 60). Cropping it.")
        data = center_crop_to_shape(data, (217, 178, 60))
    data = np.nan_to_num(data)
    data[data < 0] = 0
    data[data >= 10000] = 10000
    data = np.expand_dims(data, axis=3)
    pad = np.zeros([217, 178, 60, 1], dtype=np.float32)
    pad[: data.shape[0], : data.shape[1], : data.shape[2]] = data
    return pad
