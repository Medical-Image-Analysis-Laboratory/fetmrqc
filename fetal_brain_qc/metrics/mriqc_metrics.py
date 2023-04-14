# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# This code is a modified version of the code from the MRIQC project
# available at https://github.com/nipreps/mriqc/blob/master/mriqc/qc/anatomical.py
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
from scipy.stats import kurtosis  # pylint: disable=E0611
import numpy as np
from math import sqrt


def summary_stats(data, pvms, airmask=None, erode=True):
    r"""
    Estimates the mean, the median, the standard deviation,
    the kurtosis,the median absolute deviation (mad), the 95\%
    and the 5\% percentiles and the number of voxels (summary\_\*\_n)
    of each tissue distribution.
    .. warning ::
        Sometimes (with datasets that have been partially processed), the air
        mask will be empty. In those cases, the background stats will be zero
        for the mean, median, percentiles and kurtosis, the sum of voxels in
        the other remaining labels for ``n``, and finally the MAD and the
        :math:`\sigma` will be calculated as:
        .. math ::
            \sigma_\text{BG} = \sqrt{\sum \sigma_\text{i}^2}
    """
    from statsmodels.stats.weightstats import DescrStatsW
    from statsmodels.robust.scale import mad

    output = {}
    for label, probmap in pvms.items():
        wstats = DescrStatsW(
            data=data.reshape(-1), weights=probmap.reshape(-1)
        )
        nvox = probmap.sum()
        p05, median, p95 = wstats.quantile(
            np.array([0.05, 0.50, 0.95]),
            return_pandas=False,
        )
        thresholded = data[probmap > (0.5 * probmap.max())]

        output[label] = {
            "mean": float(wstats.mean),
            "median": float(median),
            "p95": float(p95),
            "p05": float(p05),
            "k": float(kurtosis(thresholded)),
            "stdv": float(wstats.std),
            "mad": float(mad(thresholded, center=median)),
            "n": float(nvox),
        }

    return output


def volume_fraction(pvms):
    r"""
    Computes the :abbr:`ICV (intracranial volume)` fractions
    corresponding to the (partial volume maps).
    .. math ::
        \text{ICV}^k = \frac{\sum_i p^k_i}{\sum\limits_{x \in X_\text{brain}} 1}
    :param list pvms: list of :code:`numpy.ndarray` of partial volume maps.
    """
    tissue_vfs = {}
    total = 0
    for k, seg in list(pvms.items()):
        if k == "BG":
            continue
        tissue_vfs[k] = seg.sum()
        total += tissue_vfs[k]

    for k in list(tissue_vfs.keys()):
        tissue_vfs[k] /= total
    return {k: float(v) for k, v in list(tissue_vfs.items())}


def snr(mu_fg, sigma_fg, n):
    r"""
    Calculate the :abbr:`SNR (Signal-to-Noise Ratio)`.
    The estimation may be provided with only one foreground region in
    which the noise is computed as follows:
    .. math::
        \text{SNR} = \frac{\mu_F}{\sigma_F\sqrt{n/(n-1)}},
    where :math:`\mu_F` is the mean intensity of the foreground and
    :math:`\sigma_F` is the standard deviation of the same region.
    :param float mu_fg: mean of foreground.
    :param float sigma_fg: standard deviation of foreground.
    :param int n: number of voxels in foreground mask.
    :return: the computed SNR
    """
    if n < 1 or sigma_fg == 0:
        return np.nan
    return float(mu_fg / (sigma_fg * sqrt(n / (n - 1))))


def cnr(mu_wm, mu_gm, sigma_bg, sigma_wm, sigma_gm):
    r"""
    Calculate the :abbr:`CNR (Contrast-to-Noise Ratio)` [Magnota2006]_.
    Higher values are better.
    .. math::
        \text{CNR} = \frac{|\mu_\text{GM} - \mu_\text{WM} |}{\sqrt{\sigma_B^2 +
        \sigma_\text{WM}^2 + \sigma_\text{GM}^2}},
    where :math:`\sigma_B` is the standard deviation of the noise distribution within
    the air (background) mask.
    :param float mu_wm: mean of signal within white-matter mask.
    :param float mu_gm: mean of signal within gray-matter mask.
    :param float sigma_air: standard deviation of the air surrounding the head ("hat" mask).
    :param float sigma_wm: standard deviation within white-matter mask.
    :param float sigma_gm: standard within gray-matter mask.
    :return: the computed CNR
    """
    # Does this make sense to implement this given that sigma_air=0 artificially?
    return float(
        abs(mu_wm - mu_gm)
        / sqrt(sigma_bg**2 + sigma_gm**2 + sigma_wm**2)
    )


def cjv(mu_wm, mu_gm, sigma_wm, sigma_gm):
    r"""
    Calculate the :abbr:`CJV (coefficient of joint variation)`, a measure
    related to :abbr:`SNR (Signal-to-Noise Ratio)` and
    :abbr:`CNR (Contrast-to-Noise Ratio)` that is presented as a proxy for
    the :abbr:`INU (intensity non-uniformity)` artifact [Ganzetti2016]_.
    Lower is better.
    .. math::
        \text{CJV} = \frac{\sigma_\text{WM} + \sigma_\text{GM}}{|\mu_\text{WM} - \mu_\text{GM}|}.
    :param float mu_wm: mean of signal within white-matter mask.
    :param float mu_gm: mean of signal within gray-matter mask.
    :param float sigma_wm: standard deviation of signal within white-matter mask.
    :param float sigma_gm: standard deviation of signal within gray-matter mask.
    :return: the computed CJV
    """
    if mu_wm == mu_gm:
        return np.nan
    return float((sigma_wm + sigma_gm) / abs(mu_wm - mu_gm))


def wm2max(img, mu_wm):
    r"""
    Calculate the :abbr:`WM2MAX (white-matter-to-max ratio)`,
    defined as the maximum intensity found in the volume w.r.t. the
    mean value of the white matter tissue.
    Values close to 1.0 are better:
    .. math ::
        \text{WM2MAX} = \frac{\mu_\text{WM}}{P_{99.95}(X)}
    """
    return float(mu_wm / np.percentile(img.reshape(-1), 99.95))
