""" Preprocessing for the image quality assessment based 
on FNNDSC's implementation. 

Based on the code at
https://github.com/FNNDSC/pl-fetal-brain-assessment/blob/main/fetal_brain_assessment/volume.py

 (c) 2021 Fetal-Neonatal Neuroimaging & Developmental Science Center
                       Boston Children's Hospital
    http://childrenshospital.org/FNNDSC/
                            dev@babyMRI.org
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
    import logging

    logger = logging.getLogger(__name__)
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
