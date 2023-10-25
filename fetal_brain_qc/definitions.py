import os
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
    os.path.dirname(fbqc.__file__), "data/iqms_train.csv"
)
IQMs = "/media/tsanchez/tsanchez_data/data/maddalena/iqms_full_no_test_no_avanto.csv"

NNUNET_CKPT = os.path.join(os.path.dirname(fbqc.__file__), "models/nnUNet")

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
