from fetal_brain_qc.qc_evaluation import preprocess as pp
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, SparsePCA

VALID_EXP = ["regression", "classification"]
METRICS_BASE = [
    "centroid",
    "centroid_full",
    "dl_slice_iqa_pos_only_full",
    "dl_stack_iqa_full",
    "mask_volume",
    "rank_error_center",
    "rank_error_full",
]

METRICS_BASE += [m + "_nan" for m in METRICS_BASE]
METRICS_BASE_CENTER = [
    "centroid",
    "centroid_full",
    "dl_slice_iqa",
    "dl_slice_iqa_pos_only_full",
    "dl_stack_iqa_full",
    "rank_error",
    "rank_error_center",
    "rank_error_full",
    "mask_volume",
]


METRICS_BASE_CENTER += [m + "_nan" for m in METRICS_BASE_CENTER]
METRICS = [
    "centroid",
    "centroid_full",
    "bias",
    "bias_full",
    "bias_full_not_mask",
    "dilate_erode_mask",
    "dilate_erode_mask_full",
    "dl_slice_iqa",
    "dl_slice_iqa_cropped",
    "dl_slice_iqa_full",
    "dl_slice_iqa_full_cropped",
    "dl_slice_iqa_pos_only_full",
    "dl_stack_iqa_full",
    "filter_laplace",
    "filter_laplace_full",
    "filter_laplace_mask",
    "filter_laplace_mask_full",
    "filter_sobel",
    "filter_sobel_full",
    "filter_sobel_mask",
    "filter_sobel_mask_full",
    "joint_entropy",
    "joint_entropy_full",
    "joint_entropy_intersection",
    "joint_entropy_median",
    "joint_entropy_window",
    "kurtosis",
    "kurtosis_full",
    "mae",
    "mae_window",
    "mask_volume",
    "mean",
    "mean_full",
    "median",
    "median_full",
    "mi",
    "mi_full",
    "mi_intersection",
    "mi_median",
    "mi_window",
    "ncc",
    "ncc_full",
    "ncc_intersection",
    "ncc_median",
    "ncc_window",
    "nmae",
    "nmae_window",
    "nmi",
    "nmi_full",
    "nmi_intersection",
    "nmi_median",
    "nmi_window",
    "nrmse",
    "nrmse_window",
    "percentile_5",
    "percentile_5_full",
    "percentile_95",
    "percentile_95_full",
    "psnr",
    "psnr_window",
    "rank_error",
    "rank_error_center",
    "rank_error_center_relative",
    "rank_error_full",
    "rank_error_full_cropped_relative",
    "rmse",
    "rmse_window",
    "shannon_entropy",
    "shannon_entropy_full",
    "ssim",
    "ssim_window",
    "std",
    "std_full",
    "variation",
    "variation_full",
]
METRICS += [m + "_nan" for m in METRICS]


SCALERS = [
    StandardScaler(),
    RobustScaler(),
    pp.PassThroughScaler(),
    pp.GroupRobustScaler(),
    pp.GroupStandardScaler(),
]
NOISE_FEATURES = ["passthrough", pp.NoiseWinnowFeatSelect()]
PCA_FEATURES = ["passthrough", PCA(), SparsePCA()]
