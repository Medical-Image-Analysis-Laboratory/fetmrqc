# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
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
import numpy as np
import skimage
import nibabel as ni
from fetal_brain_qc.utils import squeeze_dim


def centroid(
    mask: np.ndarray, central_third: bool = True, **kwargs
) -> np.ndarray:
    """Computes
    a motion index based on centroids of this mask. Lower is better.

    Implemented by Thomas Yu.

    Inputs
    ------
    central_third:
        whether the motion index should only be computed
        from the most central part of the data

    Output
    ------
    The computed score based on mask centroid
    """
    if central_third:
        num_z = mask.shape[2]
        center_z = int(num_z / 2.0)
        mask = mask[
            ..., int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
        ]

    centroid_coord = np.zeros((mask.shape[2], 2))
    for i in range(mask.shape[2]):
        moments = skimage.measure.moments(mask[..., i])
        centroid_coord[i, :] = [
            moments[0, 1] / moments[0, 0],
            moments[1, 0] / moments[0, 0],
        ]
    isnan = np.isnan(centroid_coord).any()
    centroid_coord = centroid_coord[~np.isnan(centroid_coord)]
    centroid_coord = np.reshape(
        centroid_coord, (int(centroid_coord.shape[0] / 2), 2)
    )
    return (
        np.var(centroid_coord[:, 0]) + np.var(centroid_coord[:, 1]),
        isnan,
    )


def mask_volume(mask, vx_size, **kwargs):
    """
    Compute volume of a nifti-encoded mask.
    Simply computes the volume of a voxel and multiply it
    by the number of voxels in the mask

    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/utilities/template_stack_estimator.py

    Input
    -----
    mask_ni:
        Nifti mask

    Output
    ------
        The volume of mask_ni (mm^3 by default)
    """

    vx_volume = np.prod(vx_size)
    isnan = False
    return np.sum(mask) * vx_volume, isnan


def rank_error(
    image,
    threshold: float = 0.99,
    central_third: bool = True,
    relative_rank: bool = True,
    **kwargs,
):
    """Given a low-resolution cropped_stack (image_cropped), computes the
    rank and svd_quality. The algorithm is based on the paper of Kainz
    et al. (2015), and ranks the stacks according to rank*svd_error,
    where lower is better.

    The implementation is based on the code in the repo NiftyMIC of Michael Ebner.

    The algorithm computes an SVD of the current stack, computes a rank r approximation
    of the original stack and iterates until the svd_error is below a given threshold.
    In Kainz' paper, they use threshold = 0.99, central_third=True

    Args:
        threshold (float, optional): Threshold for the svd_error. Defaults to 0.99.
        relative_rank (bool, optional): Whether to use the relative rank (rank/num_slices) or the absolute rank. Defaults to True.

    Returns:
        rank_error (float): The computed rank_error (rank * svd_error)
        isnan (bool): Whether the computed value is nan

    """
    if image is None:
        return np.nan, True
    # As computed in NiftyMIC
    threshold = np.sqrt(1 - threshold**2)

    if central_third:
        num_z = image.shape[2]
        center_z = int(num_z / 2.0)
        image = image[
            ..., int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
        ]

    reshaped_data = np.reshape(
        image,
        (
            image.shape[0] * image.shape[1],
            image.shape[2],
        ),
    )
    _, s, _ = np.linalg.svd(reshaped_data[:, :], full_matrices=False)
    s2 = np.square(s)
    s2_sum = np.sum(s2)

    svd_error = 2 * threshold
    rank = 0
    while svd_error > threshold:
        rank += 1
        svd_error = np.sqrt(np.sum(s2[rank:]) / s2_sum)

    isnan = False
    if relative_rank:
        # NiftyMIC: avoid penalizing stacks with a lot of slices.
        if len(s) != 0:
            rank = rank / float(len(s))
        else:
            rank = np.nan
            isnan = True

    return rank * svd_error, isnan


def normalized_cross_correlation(x, x_ref):
    """
    Compute normalized cross correlation (symmetric).

    Original code from https://github.com/gift-surg/NSoL/blob/master/nsol/similarity_measures.py

    Input
    -----
    x:
        numpy data array
    x_ref:
        reference numpy data array
    Output
    ------
    Normalized cross correlation as scalar value between -1 and 1
    """
    if x.shape != x_ref.shape:
        raise ValueError("Input data shapes do not match")

    ncc = np.sum((x - x.mean()) * (x_ref - x_ref.mean()))
    ncc /= float(x.size * x.std(ddof=1) * x_ref.std(ddof=1))

    return ncc


def shannon_entropy(x, bins=100):
    """
    Compute Shannon entropy
    Shannon entropy H(X) = - sum p(x) * ln(p(x))
    See     Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
            Mutual-information-based registration of medical images: a
            survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    Original code from https://github.com/gift-surg/NSoL/blob/master/nsol/similarity_measures.py

    Input
    -----
    x:     numpy data array
    bins:  number of bins for histogram, int

    Output
    ------
    Shannon entropy as scalar value in [0, log_b(n)]
    """
    # histogram is computed over flattened array
    hist, bin_edges = np.histogram(x, bins=bins)

    # Compute probabilities
    prob = hist / float(np.sum(hist))

    entropy = -sum([p * np.log(p) for p in prob.flatten() if p != 0])

    return entropy


def joint_entropy(x, x_ref, bins=100):
    """
    Compute joint entropy (symmetric)

    Joint entropy H(X,Y) = - sum p(x,y) * ln(p(x,y))
    See      Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
             Mutual-information-based registration of medical images: a
             survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    Date:    2017-08-04 10:35:18+0100

    Input
    -----
    x:      numpy data array
    x_ref:  reference numpy data array
    bins:   number of bins for histogram, sequence or int

    Output
    ------
    Joint entropy as scalar value >=0

    Original code from https://github.com/gift-surg/NSoL/blob/master/nsol/similarity_measures.py
    """

    hist, x_edges, y_edges = np.histogram2d(
        x.flatten(), x_ref.flatten(), bins=bins
    )

    # Compute probabilities
    prob = hist / float(np.sum(hist))

    jentropy = -sum([p * np.log(p) for p in prob.flatten() if p != 0])
    return jentropy


def mutual_information(x, x_ref, bins=100):
    """
    Compute mutual information (symmetric)

    MI(X,Y) = - sum p(x,y) * ln( p(x,y) / (p(x) * p(y)) ) = H(X) + H(Y) - H(X,Y)
    See      Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
             Mutual-information-based registration of medical images: a
             survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    See      Skouson, M.B., Quji Guo & Zhi-Pei Liang, 2001. A bound on mutual
             information for image registration. IEEE Transactions on Medical
             Imaging, 20(8), pp.843-846.
    Date:    2017-08-04 10:40:35+0100

    Input
    -----
    x:      numpy data array
    x_ref:  reference numpy data array
    bins:   number of bins for histogram, sequence or int

    Output
    ------
    Mutual information as scalar value >= 0 with upper bound as in Skouson2001

    Original code from https://github.com/gift-surg/NSoL/blob/master/nsol/similarity_measures.py
    """
    mi = shannon_entropy(x, bins=bins)
    mi += shannon_entropy(x_ref, bins=bins)
    mi -= joint_entropy(x, x_ref, bins=bins)
    return mi


def normalized_mutual_information(x, x_ref, bins=100):
    """
    Compute mutual information (symmetric)

    NMI(X,Y) = H(X) + H(Y) / H(X,Y)
    See         Pluim, J.P.W., Maintz, J.B.A. & Viergever, M.A., 2003.
                Mutual-information-based registration of medical images: a
                survey. IEEE Transactions on Medical Imaging, 22(8), pp.986-1004.
    Date:       2017-08-04 10:40:35+0100

    Input
    -----
    x:      numpy data array
    x_ref:  reference numpy data array
    bins:   number of bins for histogram, sequence or int

    Output
    ------
    Normalized mutual information as scalar value >= 0

    Original code from https://github.com/gift-surg/NSoL/blob/master/nsol/similarity_measures.py
    """
    nmi = shannon_entropy(x, bins=bins)
    nmi += shannon_entropy(x_ref, bins=bins)
    nmi /= joint_entropy(x, x_ref, bins=bins)
    return nmi


def psnr(x, x_ref, datarange=None, **kwargs):
    if not datarange:
        datarange = int(np.amax(x_ref) - min(np.amin(x), np.amin(x_ref)))

    if sum(abs(x - x_ref)) < 1e-13:
        # Avoiding to compute the psnr on exactly the same slices.
        return np.nan
    psnr = skimage.metrics.peak_signal_noise_ratio(
        x, x_ref, data_range=datarange
    )
    return psnr


def nrmse(x, x_ref, **kwargs):
    return skimage.metrics.normalized_root_mse(x, x_ref)


def rmse(x, x_ref):
    return np.sqrt(skimage.metrics.mean_squared_error(x, x_ref))


def mae(x, x_ref):
    return np.mean(abs(x - x_ref))


def nmae(x, x_ref):
    return mae(x, x_ref) / np.mean(np.abs(x_ref))


def ssim(x, x_ref, mask=None, datarange=None):
    if not datarange:
        datarange = int(np.amax(x_ref) - min(np.amin(x), np.amin(x_ref)))

    # SSIM
    ssim = skimage.metrics.structural_similarity(
        x_ref,
        x,
        data_range=datarange,
        full=True,
    )

    def pick_ssim(ssim, mask):
        """Pick the SSIM output:
        1- If masked input, take the full SSIM on the masked input
        2- Else, return the average SSIM
        """
        if mask is None:
            return ssim[0]
        else:
            return ssim[1][mask > 0].mean()

    return pick_ssim(ssim, mask)
