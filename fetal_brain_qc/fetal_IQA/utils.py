from fetal_brain_utils.cropping import crop_image_to_region
from copy import deepcopy
import numpy as np
import nibabel as ni
import torch
from pathlib import Path


def eval_model(image, mask, model, device):
    try:
        image, mask = adjust_around_mask_to_256(image, mask)
        zmin = min(np.where(mask == 1)[2])
    except Exception as e:
        print(f"WARNING: Failed mask extraction: {e}")
        return None
    img = np.transpose(image, [2, 0, 1])

    # the size of image should be 256x256
    assert img.shape[1] == 256 and img.shape[2] == 256

    # Standardize
    img = (img - np.mean(img)) / np.std(img)

    # the input to the network should have 3 channels
    img = np.stack((img, img, img), axis=1)

    # predict
    with torch.no_grad():
        pred, _ = model(torch.tensor(img, dtype=torch.float32, device=device))
        pred = torch.softmax(pred, dim=1).cpu().numpy()

    # out of ROI is also captured by prob. low-quality.
    # We simply do sum of prob to be good - prob to be bad / total number of slices
    weighted_stack = (pred[:, 1].sum() - pred[:, 2].sum()) / pred.shape[0]
    pred_dict = {
        i
        + zmin: {
            "in": pred[i, 0],
            "good": pred[i, 1],
            "bad": pred[i, 2],
            "mean_in": pred[:, 0].mean(),
            "mean_good": pred[:, 1].mean(),
            "mean_bad": pred[:, 2].mean(),
            "weighted": weighted_stack,
        }
        for i in range(img.shape[0])
    }
    return pred_dict


def get_256_range(im_size, mask_center):
    """
    Given a image size (`im_size`) and the position of the centroid
    (`mask_center`), computes that puts the brain mask as much as
    possible to the center of the image while keeping a 256 image size.

    """
    if im_size < 256:
        print("WARNING: Axis is smaller than 256.")
        return [0, im_size]
    assert (
        mask_center <= im_size
    ), "The mask centroid is larger than the image shape"
    crop_range_low = mask_center - 128
    crop_range_high = mask_center + 128
    if crop_range_low < 0:
        crop_range_high += -crop_range_low
        crop_range_low += -crop_range_low

    if crop_range_high > im_size:
        crop_range_low -= crop_range_high - im_size
        crop_range_high -= crop_range_high - im_size
    return crop_range_low, crop_range_high


def pad_image(data, target):
    """Pads the image to the target size (not centered)"""
    target = np.zeros(target)
    x, y, z = data.shape
    target[:x, :y, :z] = data
    return target


def crop_pad_img(data, xrange, yrange, zrange):
    """Utility function to apply the cropping around the brain,
    pad the image if it is too small and return a nifti file.
    """
    data = crop_image_to_region(data, xrange, yrange, zrange)
    if any(x < 256 for x in data.shape[:2]):
        pad_to = (256, 256, data.shape[2])
        data = pad_image(data, pad_to)
    return data


def adjust_around_mask_to_256(image, mask):
    """Crop the image and mask to a 256x256 in-plane resolution
    based on the brain mask centroid across all the stack. Also pads
    the image if needed. The method ensures that the returned image is of
    size 256 also when the brain isn't centered in a stack.
    """
    image = image.transpose(2, 1, 0)
    mask = mask.transpose(2, 1, 0)
    coords = np.where(mask == 1)

    # Discard empty masks
    if len(coords[0]) == 0:
        raise ValueError("Empty mask.")

    xmean, ymean, = (
        int(coords[0].mean()),
        int(coords[1].mean()),
    )
    xshape, yshape = mask.shape[:2]
    xrange = get_256_range(xshape, xmean)
    yrange = get_256_range(yshape, ymean)
    zrange = (min(coords[2]), max(coords[2] + 1))

    crop_fct = lambda im: crop_pad_img(im, xrange, yrange, zrange)

    return crop_fct(image), crop_fct(mask)
