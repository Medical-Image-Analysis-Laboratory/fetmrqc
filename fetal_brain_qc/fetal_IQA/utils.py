from fetal_brain_qc.utils import crop_image_to_region
from copy import deepcopy
import numpy as np
import nibabel as ni
import torch
from pathlib import Path


def eval_model(im_path, mask_path, model, device):
    """Evaluates the fetal IQA model:
    1. Crop the input image to 256x256 based on the brain mask
    2. Run the evaluation and return a slice-wise score as well as global ratings.
    """
    from fetal_brain_qc.fetal_IQA.utils import adjust_around_mask_to_256

    image_ni = ni.load(im_path)
    mask_ni = ni.load(mask_path)

    print(f"Processing {Path(im_path).name}")
    try:
        image_ni, mask_ni = adjust_around_mask_to_256(image_ni, mask_ni)
        zmin = min(np.where(mask_ni.get_fdata() == 1)[2])
    except RuntimeError as e:
        print(f"{Path(im_path).name}: {e}")
        pred_dict = {
            0: {
                "in": None,
                "good": None,
                "bad": None,
                "mean_in": None,
                "mean_good": None,
                "mean_bad": None,
                "weighted": None,
            }
        }
        return pred_dict

    img = np.transpose(image_ni.get_fdata(), [2, 0, 1])

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


def crop_pad_nifti(data, affine, xrange, yrange, zrange):
    """Utility function to apply the cropping around the brain,
    pad the image if it is too small and return a nifti file.
    """
    print("input", data.shape, xrange, yrange, zrange)
    data = crop_image_to_region(data, xrange, yrange, zrange)
    print("before", data.shape)
    if any(x < 256 for x in data.shape[:2]):
        pad_to = (256, 256, data.shape[2])
        data = pad_image(data, pad_to)
    print(data.shape)
    return ni.Nifti1Image(data, affine)


def adjust_around_mask_to_256(image_ni, mask_ni):
    """Crop the image and mask to a 256x256 in-plane resolution
    based on the brain mask centroid across all the stack. Also pads
    the image if needed. The method ensures that the returned image is of
    size 256 also when the brain isn't centered in a stack.
    """

    from fetal_brain_qc.utils import crop_image_to_region
    from copy import deepcopy

    image = image_ni.get_fdata().squeeze()
    mask = mask_ni.get_fdata().squeeze()
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

    new_origin = list(
        ni.affines.apply_affine(
            mask_ni.affine, [xrange[0], yrange[0], yrange[0]]
        )
    ) + [1]
    image_ni = deepcopy(image_ni)
    mask_ni = deepcopy(mask_ni)

    new_affine = image_ni.affine
    new_affine[:, -1] = new_origin
    crop_fct = lambda im: crop_pad_nifti(
        im, new_affine, xrange, yrange, zrange
    )

    return crop_fct(image), crop_fct(mask)
