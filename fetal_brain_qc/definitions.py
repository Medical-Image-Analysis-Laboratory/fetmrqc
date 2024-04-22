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
import os
from pathlib import Path
import fetal_brain_qc as fbqc


MANU_BASE = (
    "/media/tsanchez/tsanchez_data/data/data_anon/derivatives/refined_masks/"
)
AUTO_BASE = (
    "/media/tsanchez/tsanchez_data/data/data_anon/derivatives/automated_masks/"
)
MASK_PATTERN = (
    "sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
    "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz"
)

MASK_PATTERN_LIST = [MANU_BASE + MASK_PATTERN, AUTO_BASE + MASK_PATTERN]

BRAIN_CKPT = os.path.join(
    os.path.dirname(fbqc.__file__), "models/MONAIfbs_dynunet_ckpt.pt"
)

FETAL_IQA_CKPT = os.path.join(
    os.path.dirname(fbqc.__file__), "models/fetal_IQA_pytorch.ckpt"
)

FETAL_STACK_IQA_CKPT = os.path.join(
    os.path.dirname(fbqc.__file__), "models/FNNDSC_qcnet_ckpt.hdf5"
)

FETAL_FETMRQC_CLF_CKPT = os.path.join(
    os.path.dirname(fbqc.__file__), "models/fetmrqc_full_class.joblib"
)
FETAL_FETMRQC_REG_CKPT = os.path.join(
    os.path.dirname(fbqc.__file__), "models/fetmrqc_full_regr.joblib"
)
FETAL_FETMRQC20_CLF_CKPT = os.path.join(
    os.path.dirname(fbqc.__file__), "models/fetmrqc_20_class.joblib"
)
FETAL_FETMRQC20_REG_CKPT = os.path.join(
    os.path.dirname(fbqc.__file__), "models/fetmrqc_20_regr.joblib"
)
IQMs = os.path.join(os.path.dirname(fbqc.__file__), "data/iqms_train.csv")

NNUNET_CKPT = os.path.join(os.path.dirname(fbqc.__file__), "models/nnUNet")
NNUNET_ENV_DEFAULT = Path(os.getenv("CONDA_PREFIX")).parent / "nnunet"

FETMRQC20 = [
    "rank_error",
    "dilate_erode_mask_full",
    "mask_volume",
    "filter_sobel_mask_full",
    "nrmse_window",
    "filter_laplace_mask",
    "filter_laplace_mask_full",
    "dilate_erode_mask",
    "rank_error_center",
    "seg_sstats_BG_n",
    "centroid",
    "rank_error_center_relative",
    "seg_sstats_CSF_n",
    "seg_sstats_GM_n",
    "im_size_z",
    "ncc_intersection",
    "ncc_window",
    "psnr_window",
    "seg_snr_WM",
    "seg_volume_GM",
]
# This is a slightly different list from FETMRQC20
# as some metrics are only available in groups
FETMRQC20_METRICS = [
    "rank_error",
    "dilate_erode_mask_full",
    "mask_volume",
    "filter_sobel_mask_full",
    "nrmse_window",
    "filter_laplace_mask",
    "filter_laplace_mask_full",
    "dilate_erode_mask",
    "rank_error_center",
    "centroid",
    "rank_error_center_relative",
    "seg_sstats",
    "im_size",
    "ncc_intersection",
    "ncc_window",
    "psnr_window",
    "seg_snr",
    "seg_volume",
]
