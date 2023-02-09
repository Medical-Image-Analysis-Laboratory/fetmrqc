import numpy as np
import nibabel as ni
import skimage
from .utils import (
    allow_kwargs,
    freeze,
    normalized_cross_correlation,
    shannon_entropy,
    joint_entropy,
    mutual_information,
    normalized_mutual_information,
    psnr,
    rmse,
    mae,
    nmae,
    nrmse,
    ssim,
)
from skimage.morphology import binary_dilation, binary_erosion
from skimage.filters import sobel, laplace
from inspect import getmembers, isfunction
from fetal_brain_utils import get_cropped_stack_based_on_mask
from fetal_brain_qc.fnndsc_IQA import fnndsc_preprocess
from fetal_brain_qc.utils import squeeze_dim
from scipy.stats import kurtosis, variation
from functools import partial
import pandas as pd


SKIMAGE_FCT = [fct for _, fct in getmembers(skimage.filters, isfunction)]
DEFAULT_METRICS = [
    "dl_slice_iqa_full",
    "dl_stack_iqa_full",
    "centroid",
    "rank_error",
    "mask_volume",
    "ncc",
    "nmi",
]
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class LRStackMetrics:
    """Contains a battery of metrics that can be evaluated on individual
    pairs of low-resolution stacks and masks. The various combinations possible
    are declared in `metrics_func`, which currently contains:
        * metrics based on `_metric_mask_centroid`: computes the centroid of a
            brain_mask at each slice and returns an index based on its
            displacement.
            Variants inlucde:
                * centroid_center: consider only the third of more central
                    slices
                * centroid_full: consider all slices
        * metrics based on  `_metric_rank_error`: Method proposed in the TMI
            paper of Kainz et al. (2015), estimating motion by how easily the
            LR stack can be approximated by a low-rank approximation.
            Variants include:
            * _center/_full: whether only the central third slices should be
                considered.
            * _cropped/_full: whether the LR image should be cropped based on
                the bounding box around the mask (done by Ebner et al. (2020)
                in NiftyMIC)
            * _relative/_no-relative: whether the computation of the index
            should include a relative rank (rank/nstacks) or simply be the
            rank (proposed by Ebner et al. (2020) in NiftyMIC)
        * `_mask_volume`: Compute the volume of the mask in mm^3
    """

    def __init__(
        self,
        metrics=None,
        ckpt_stack_iqa=None,
        ckpt_slice_iqa=None,
        device=None,
    ):
        default_params = dict(
            central_third=True,
            crop_image=True,
            compute_on_mask=True,
            mask_intersection=False,
            reduction="mean",
        )
        self._metrics = self.get_default_metrics() if not metrics else metrics
        self.stack_iqa_enabled = True if ckpt_stack_iqa else False
        if ckpt_stack_iqa:
            import os

            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            from fetal_brain_qc.fnndsc_IQA import Predictor

            self.stack_predictor = Predictor(ckpt_stack_iqa)
        self.slice_iqa_enabled = True if ckpt_slice_iqa and device else False
        if ckpt_slice_iqa:
            from fetal_brain_qc.fetal_IQA import resnet34
            import torch

            self.device = device
            self.slice_model = resnet34(pretrained=False, num_classes=3)
            self.slice_model = torch.nn.DataParallel(self.slice_model).to(
                device
            )
            checkpoint = torch.load(ckpt_slice_iqa, map_location=device)
            self.slice_model.load_state_dict(checkpoint["ema_state_dict"])
            self.slice_model.eval()

        self.metrics_func = {
            "centroid": freeze(self._metric_mask_centroid, central_third=True),
            "centroid_full": freeze(
                self._metric_mask_centroid, central_third=False
            ),
            "rank_error": freeze(
                self._metric_rank_error,
                threshold=0.99,
                central_third=True,
                crop_image=True,
                relative_rank=True,
            ),
            "rank_error_full": freeze(
                self._metric_rank_error,
                threshold=0.99,
                central_third=False,
                crop_image=False,
                relative_rank=False,
            ),
            "rank_error_center": freeze(
                self._metric_rank_error,
                threshold=0.99,
                central_third=True,
                crop_image=False,
                relative_rank=False,
            ),
            "rank_error_center_relative": freeze(
                self._metric_rank_error,
                threshold=0.99,
                central_third=True,
                crop_image=False,
                relative_rank=True,
            ),
            "rank_error_full_cropped_relative": freeze(
                self._metric_rank_error,
                central_third=False,
                crop_image=True,
                relative_rank=True,
            ),
            "rank_error_center": freeze(
                self._metric_rank_error,
                central_third=True,
                crop_image=False,
                relative_rank=False,
            ),
            "mask_volume": self._metric_mask_volume,
            "ncc": self.process_metric(
                metric=normalized_cross_correlation, **default_params
            ),
            "ncc_window": self.process_metric(
                metric=normalized_cross_correlation,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "ncc_median": self.process_metric(
                metric=normalized_cross_correlation,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "ncc_intersection": self.process_metric(
                metric=normalized_cross_correlation,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "ncc_full": self.process_metric(
                metric=normalized_cross_correlation,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "joint_entropy": self.process_metric(
                metric=joint_entropy, **default_params
            ),
            "joint_entropy_window": self.process_metric(
                metric=joint_entropy,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "joint_entropy_median": self.process_metric(
                metric=joint_entropy,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "joint_entropy_intersection": self.process_metric(
                metric=joint_entropy,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "joint_entropy_full": self.process_metric(
                metric=joint_entropy,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "mi": self.process_metric(
                metric=mutual_information, **default_params
            ),
            "mi_window": self.process_metric(
                metric=mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "mi_median": self.process_metric(
                metric=mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "mi_intersection": self.process_metric(
                metric=mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "mi_full": self.process_metric(
                metric=mutual_information,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "nmi": self.process_metric(
                metric=normalized_mutual_information, **default_params
            ),
            "nmi_window": self.process_metric(
                metric=normalized_mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "nmi_median": self.process_metric(
                metric=normalized_mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "nmi_intersection": self.process_metric(
                metric=normalized_mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "nmi_full": self.process_metric(
                metric=normalized_mutual_information,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "shannon_entropy": self.process_metric(
                shannon_entropy,
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "shannon_entropy_full": self.process_metric(
                shannon_entropy,
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            # This should be the default metric, but
            # this preprocessing does not work with stack iqa
            # "dl_stack_iqa": self.process_metric(
            #     self._metric_stack_iqa,
            #     type="dl",
            #     central_third=True,
            #     crop_image=True,
            # ),
            "dl_stack_iqa_full": self.process_metric(
                self._metric_stack_iqa,
                type="dl",
                central_third=False,
                crop_image=True,
            ),
            "dl_slice_iqa": self.process_metric(
                self._metric_slice_iqa,
                type="dl",
                central_third=True,
                crop_image=False,
            ),
            "dl_slice_iqa_cropped": self.process_metric(
                self._metric_slice_iqa,
                type="dl",
                central_third=True,
                crop_image=True,
            ),
            "dl_slice_iqa_full": self.process_metric(
                self._metric_slice_iqa,
                type="dl",
                central_third=False,
                crop_image=False,
            ),
            "dl_slice_iqa_full_cropped": self.process_metric(
                self._metric_slice_iqa,
                type="dl",
                central_third=False,
                crop_image=True,
            ),
            "dl_slice_iqa_pos_only_full": self.process_metric(
                self._metric_slice_iqa,
                type="dl",
                central_third=False,
                crop_image=False,
                positive_only=True,
            ),
            "psnr": self.process_metric(
                psnr,
                use_datarange=True,
            ),
            "psnr_window": self.process_metric(
                psnr,
                use_datarange=True,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "nrmse": self.process_metric(nrmse, **default_params),
            "nrmse_window": self.process_metric(
                nrmse,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "rmse": self.process_metric(rmse, **default_params),
            "rmse_window": self.process_metric(
                rmse,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "nmae": self.process_metric(nmae, **default_params),
            "nmae_window": self.process_metric(
                nmae,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "mae": self.process_metric(mae, **default_params),
            "mae_window": self.process_metric(
                mae,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "ssim": freeze(self._ssim, **default_params),
            "ssim_window": freeze(
                self._ssim,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="mean",
                use_window=True,
            ),
            "mean": self.process_metric(
                np.mean,
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "mean_full": self.process_metric(
                np.mean,
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            "std": self.process_metric(
                np.std,
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "std_full": self.process_metric(
                np.std,
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            "median": self.process_metric(
                np.median,
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "median_full": self.process_metric(
                np.median,
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            "percentile_5": self.process_metric(
                partial(np.percentile, q=5),
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "percentile_5_full": self.process_metric(
                partial(np.percentile, q=5),
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            "percentile_95": self.process_metric(
                partial(np.percentile, q=95),
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "percentile_95_full": self.process_metric(
                partial(np.percentile, q=95),
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            "kurtosis": self.process_metric(
                kurtosis,
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "kurtosis_full": self.process_metric(
                kurtosis,
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            "variation": self.process_metric(
                variation,
                type="noref",
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
            ),
            "variation_full": self.process_metric(
                variation,
                type="noref",
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
            ),
            "bias": freeze(
                self._metric_bias_field,
                compute_on_mask=True,
                central_third=True,
            ),
            "bias_full": freeze(
                self._metric_bias_field,
                compute_on_mask=True,
                central_third=False,
            ),
            "bias_full_not_mask": freeze(
                self._metric_bias_field,
                compute_on_mask=False,
                central_third=False,
            ),
            ## Filter-based metrics
            "dilate_erode_mask": freeze(
                self._metric_dilate_erode_mask, central_third=True
            ),
            "dilate_erode_mask_full": freeze(
                self._metric_dilate_erode_mask, central_third=False
            ),
            "filter_laplace_mask": freeze(
                self._metric_filter_mask, filter=laplace
            ),
            "filter_laplace_mask_full": freeze(
                self._metric_filter_mask, filter=laplace, central_third=False
            ),
            "filter_sobel_mask": freeze(
                self._metric_filter_mask, filter=sobel
            ),
            "filter_sobel_mask_full": freeze(
                self._metric_filter_mask, filter=sobel, central_third=False
            ),
            "filter_laplace": freeze(self._metric_filter, filter=laplace),
            "filter_laplace_full": freeze(
                self._metric_filter, filter=laplace, central_third=False
            ),
            "filter_sobel": freeze(self._metric_filter, filter=sobel),
            "filter_sobel_full": freeze(
                self._metric_filter, filter=sobel, central_third=False
            ),
        }

        self._check_metrics()
        self.normalization = None
        self.norm_dict = {}

    def get_default_metrics(self):
        return DEFAULT_METRICS

    def get_all_metrics(self):
        return list(self.metrics_func.keys())

    def set_metrics(self, metrics):
        self._metrics = metrics

    def _valid_mask(self, mask_path):
        mask = ni.load(mask_path).get_fdata()
        if mask.sum() == 0:
            return False
        else:
            return True

    def _max_normalize(self, group):
        """Normalize the data to have on average the max at 255."""
        max_list = []
        iter_ = (
            group.iterrows() if not isinstance(group, pd.Series) else [group]
        )
        for row in iter_:
            row = row[1] if not isinstance(group, pd.Series) else row
            im, _ = self._load_and_prep_nifti(
                row["im"], row["mask"], crop_image=True, central_third=False
            )
            assert (im >= 0).all(), "Images should have positive values."
            max_list.append(np.max(im))
        factor = 255 / np.mean(max_list)
        return factor

    def normalize_dataset(self, bids_list, normalization="sub_ses"):
        """Taking a `bids_list` obtained using csv_to_list,
        computes the factor that should be used to scale the input to the LR method."""

        assert normalization in ["sub_ses", "site", "run", None]

        # This is not the cleanest: normalization needs to be None because
        # the _max_normalize functions calls _load_and_prep_nifti
        # which uses the computed normalization.
        self.normalization = None
        self.norm_dict = {}
        df = pd.DataFrame(bids_list)

        if normalization is None:
            self.norm_dict = None
            return
        elif normalization in ["sub_ses", "site"]:
            grp_by = ["sub", "ses"] if normalization == "sub_ses" else "site"
            grp = (
                df.groupby(grp_by, group_keys=True)
                .apply(self._max_normalize)
                .rename("norm")
            )
            df = (
                df.set_index(grp_by)
                .merge(grp, left_index=True, right_index=True)
                .reset_index()
            )
        else:
            df["norm"] = df.apply(self._max_normalize, axis=1)
        self.normalization = normalization
        self.norm_dict = df[["im", "norm"]].set_index("im").to_dict()["norm"]

    def evaluate_metrics(self, lr_path, mask_path):
        """TODO"""
        # Remark: Could do something better with a class here: giving flexible
        # features as input.
        args_dict = {"lr_path": lr_path, "mask_path": mask_path}
        results = {}
        if not self._valid_mask(mask_path):
            print(f"\tWARNING: Empty mask {mask_path}.")
        for m in self._metrics:
            print("\tRunning", m)
            if self._valid_mask(mask_path):
                out = self.metrics_func[m](**args_dict)
            else:
                out = [None, None]
            results[m], results[m + "_nan"] = out
        return results

    def _check_metrics(self):
        """TODO"""
        for m in self._metrics:
            if m not in self.metrics_func.keys():
                raise RuntimeError(
                    f"Metric {m} is not part of the available metrics."
                )
        if "dl_stack_iqa" in self._metrics:
            assert (
                self.stack_iqa_enabled
            ), "dl_stack_iqa requires passing a checkpoint for for the DL stack-wise model."
        if "dl_slice_iqa" in self._metrics:
            assert (
                self.slice_iqa_enabled
            ), "dl_slice_iqa requires passing a checkpoint and a device for for the DL slice-wise model."

    def process_metric(
        self,
        metric,
        type="ref",
        **kwargs,
    ):
        if type == "ref":
            return freeze(
                self.preprocess_and_evaluate_metric, metric=metric, **kwargs
            )
        elif type == "noref":
            return freeze(
                self.preprocess_and_evaluate_noref_metric,
                noref_metric=metric,
                **kwargs,
            )
        elif type == "dl":
            return freeze(
                self.preprocess_and_evaluate_dl_metric,
                dl_metric=metric,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Unknown metric type {type}. Please choose among ['ref', 'noref', 'dl']"
            )

    @allow_kwargs
    def _metric_mask_centroid(
        self, mask_path: str, central_third: bool = True
    ) -> np.ndarray:
        """Given a path to a brain mask `mask_path`, computes
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
        mask_ni = ni.load(mask_path)
        mask = squeeze_dim(mask_ni.get_fdata(), -1)
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

    @allow_kwargs
    def _metric_mask_volume(self, mask_path):
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

        mask_ni = ni.load(mask_path)
        mask = squeeze_dim(mask_ni.get_fdata(), -1)
        vx_volume = np.array(mask_ni.header.get_zooms()).prod()
        isnan = False
        return np.sum(mask) * vx_volume, isnan

    @allow_kwargs
    def _metric_rank_error(
        self,
        lr_path,
        mask_path,
        threshold: float = 0.99,
        central_third: bool = True,
        crop_image: bool = True,
        relative_rank: bool = True,
    ) -> np.ndarray:
        """Given a low-resolution cropped_stack (image_cropped), computes the
        rank and svd_quality. The algorithm is based on the paper of Kainz
        et al. (2015), and ranks the stacks according to rank*svd_error,
        where lower is better.

        The implementation is based on the code in the repo NiftyMIC of Michael Ebner.

        The algorithm computes an SVD of the current stack, computes a rank r approximation
        of the original stack and iterates until the svd_error is below a given threshold.
        In Kainz' paper, they use threshold = 0.99, central_third=True

        Inputs
        ------
        lr_path:
        mask_path:
        threshold:
        central_third:
        crop_image:
        relative_rank:
        Output
        ------
        """
        image_ni = ni.load(lr_path)
        image = image_ni.get_fdata()
        mask_ni = ni.load(mask_path)
        mask = squeeze_dim(mask_ni.get_fdata(), -1)

        if crop_image:
            image = get_cropped_stack_based_on_mask(image_ni, mask_ni)
            if image is None:
                return np.nan, True
            image = image.get_fdata()
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

    def _load_and_prep_nifti(
        self, lr_path, mask_path, crop_image, central_third
    ):
        """TODO"""
        image_ni = ni.load(lr_path)
        image = squeeze_dim(image_ni.get_fdata(), -1).transpose(2, 1, 0)
        mask_ni = ni.load(mask_path)
        mask = squeeze_dim(mask_ni.get_fdata(), -1).transpose(2, 1, 0)
        if mask.sum() == 0.0:
            return None, None

        if crop_image:
            image = get_cropped_stack_based_on_mask(image_ni, mask_ni)
            mask = get_cropped_stack_based_on_mask(mask_ni, mask_ni)
            if image is None or mask is None:
                return None, None
            image = squeeze_dim(image.get_fdata(), -1).transpose(2, 1, 0)
            mask = squeeze_dim(mask.get_fdata(), -1).transpose(2, 1, 0)
        if central_third:
            num_z = image.shape[0]
            center_z = int(num_z / 2.0)
            image = image[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0) + 1
            ]
            mask = mask[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0) + 1
            ]

        if self.normalization is not None:
            norm = self.norm_dict[lr_path]
            image = image * norm
        return image, mask

    def _ssim(
        self,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
        use_window=False,
        window_size=3,
    ):
        image, mask = self._load_and_prep_nifti(
            lr_path, mask_path, crop_image, central_third
        )

        if image is None or mask is None:
            # image is None when the mask is empty: nothing is computed.
            return np.nan, True
        metric_out = []
        isnan = False
        datarange = image[mask > 0].max() - image[mask > 0].min()

        for i, img_i in enumerate(image):
            if use_window:
                l, r = window_size // 2, window_size - window_size // 2
                range_j = range(max(0, i - l), min(image.shape[0], i + r))
            else:
                range_j = range(0, image.shape[0])
            for j in range_j:
                im_i = img_i
                im_j = image[j]
                mask_curr = (
                    mask[i] * mask[j]
                    if mask_intersection
                    else ((mask[i] + mask[j]) > 0).astype(int)
                )

                m = (
                    ssim(im_i, im_j, mask_curr, datarange)
                    if compute_on_mask
                    else ssim(im_i, im_j, datarange)
                )
                # Try to not consider self-correlation
                if not np.isnan(m) and i != j:
                    metric_out.append(m)
                if np.isnan(m):
                    isnan = True

        if reduction == "mean":
            return np.mean(metric_out), isnan
        elif reduction == "median":
            return np.median(metric_out), isnan

    def preprocess_and_evaluate_metric(
        self,
        metric,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
        use_window=False,
        window_size=3,
        use_datarange=False,
    ):

        VALID_REDUCTIONS = ["mean", "median"]
        assert reduction in VALID_REDUCTIONS, (
            f"Unknown reduction function {reduction}."
            f"Choose from {VALID_REDUCTIONS}"
        )
        image, mask = self._load_and_prep_nifti(
            lr_path, mask_path, crop_image, central_third
        )
        if image is None or mask is None:
            # image is None when the mask is empty: nothing is computed.
            return np.nan, True
        metric_out = []
        isnan = False
        if use_datarange:
            if compute_on_mask:
                datarange = image[mask > 0].max() - image[mask > 0].min()
            else:
                datarange = image.max() - image.min()
        for i, img_i in enumerate(image):
            if use_window:
                l, r = window_size // 2, window_size - window_size // 2
                range_j = range(max(0, i - l), min(image.shape[0], i + r))
            else:
                range_j = range(0, image.shape[0])
            for j in range_j:
                im_i = img_i
                im_j = image[j]
                if compute_on_mask:
                    idx = (
                        np.where(mask[i] * mask[j])
                        if mask_intersection
                        else np.where(mask[i] + mask[j])
                    )
                    im_i, im_j = im_i[idx], im_j[idx]
                if use_datarange:
                    m = metric(im_i, im_j, datarange)
                else:
                    m = metric(im_i, im_j)

                if not np.isnan(m) and i != j:
                    metric_out.append(m)
                if np.isnan(m):
                    isnan = True
        if reduction == "mean":
            return np.mean(metric_out), isnan
        elif reduction == "median":
            return np.median(metric_out), isnan

    def preprocess_and_evaluate_noref_metric(
        self,
        noref_metric,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        flatten=True,
    ):
        image, mask = self._load_and_prep_nifti(
            lr_path, mask_path, crop_image, central_third
        )
        if compute_on_mask:
            image = image[np.where(mask)]
        if flatten:
            metric = noref_metric(image.flatten())
        return metric, np.isnan(metric)

    def preprocess_and_evaluate_dl_metric(
        self,
        dl_metric,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        positive_only=False,
    ):
        image, mask = self._load_and_prep_nifti(
            lr_path, mask_path, crop_image, central_third
        )

        metric = dl_metric(image, mask, positive_only=False)
        return metric, np.isnan(metric)

    @allow_kwargs
    def _metric_stack_iqa(self, image, mask, positive_only=None) -> np.ndarray:
        """ """
        # Loading data

        # Input to fnndsc must be n_h x n_w x n_slices, not the other way around.
        image = image.transpose(2, 1, 0)
        mask = mask.transpose(2, 1, 0)
        image = fnndsc_preprocess(image, mask)
        df = self.stack_predictor.predict([image], ["img"])
        df = df.set_index("filename")
        return df.loc["img"]["quality"]

    @allow_kwargs
    def _metric_slice_iqa(
        self,
        image,
        mask,
        positive_only=False,
    ) -> np.ndarray:
        """ """
        # Loading data
        from fetal_brain_qc.fetal_IQA import eval_model

        iqa_dict = eval_model(image, mask, self.slice_model, self.device)
        if iqa_dict is None:
            return np.nan
        weighted_score = iqa_dict[list(iqa_dict.keys())[0]]["weighted"]
        p_good, p_bad = [], []
        for v in iqa_dict.values():
            p_good.append(v["good"])
            p_bad.append(v["bad"])

        if positive_only is None:
            weighted_score = sum(p_good) / len(p_good)
        else:
            weighted_score = (sum(p_good) - sum(p_bad)) / len(p_good)
        return weighted_score

    @allow_kwargs
    def _metric_bias_field(
        self,
        lr_path,
        mask_path,
        compute_on_mask=True,
        central_third=True,
        spline_order=3,
        wiener_filter_noise=0.11,
        convergence_threshold=1e-6,
        fwhm=0.15,
    ) -> np.ndarray:
        """ """

        import SimpleITK as sitk

        bias_corr = sitk.N4BiasFieldCorrectionImageFilter()

        bias_corr.SetBiasFieldFullWidthAtHalfMaximum(fwhm)
        bias_corr.SetConvergenceThreshold(convergence_threshold)
        bias_corr.SetSplineOrder(spline_order)
        bias_corr.SetWienerFilterNoise(wiener_filter_noise)

        image_sitk = sitk.ReadImage(str(lr_path), sitk.sitkFloat64)

        sitk_mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        # Allows to deal with masks that have a different shape than the input image.
        sitk_mask = sitk.Resample(
            sitk_mask,
            image_sitk,
            sitk.Euler3DTransform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk_mask.GetPixelIDValue(),
        )
        sitk_mask.CopyInformation(image_sitk)
        bias_corr.Execute(image_sitk, sitk_mask)
        bias_field = sitk.Cast(
            sitk.Exp(bias_corr.GetLogBiasFieldAsImage(image_sitk)),
            sitk.sitkFloat64,
        )
        bias_error = sitk.GetArrayFromImage(
            abs(image_sitk - image_sitk / bias_field)
        )
        im_ref = sitk.GetArrayFromImage(image_sitk)
        mask = sitk.GetArrayFromImage(sitk_mask)
        if central_third:
            num_z = im_ref.shape[0]
            center_z = int(num_z / 2.0)
            bias_error = bias_error[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]
            im_ref = im_ref[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]
            mask = mask[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]
        if compute_on_mask:
            bias_error = bias_error[mask > 0]
            im_ref = im_ref[mask > 0]

        isnan = np.any(np.isnan(bias_error))
        bias_nmae = np.nanmean(bias_error) / np.nanmean(abs(im_ref))
        return bias_nmae, isnan

    ### Filter-based metrics

    @allow_kwargs
    def _metric_dilate_erode_mask(
        self, mask_path: str, central_third: bool = True
    ) -> np.ndarray:
        """Given a path to a brain mask `mask_path`, dilates and
        erodes the mask in the z-direction to see the overlap between masks
        in consecutive slices.

        Inputs
        ------
        central_third:
            whether the dilation-erosion should only be computed
            from the most central part of the data

        Output
        ------
        """

        # Structuring element (Vertical line to only dilate and erode)
        # through the plane
        struc = np.array(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]
        )

        mask_ni = ni.load(mask_path)
        mask = squeeze_dim(mask_ni.get_fdata(), -1)

        if central_third:
            num_z = mask.shape[2]
            center_z = int(num_z / 2.0)
            mask = mask[
                ..., int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]
        processed = binary_erosion(binary_dilation(mask, struc), struc)
        volume = np.sum(mask)
        if volume == 0.0:
            return np.nan, True
        else:
            res = np.sum(abs(processed - mask)) / volume
            return res, False

    @allow_kwargs
    def _metric_filter_mask(
        self, mask_path: str, filter=None, central_third: bool = True
    ) -> np.ndarray:
        """Given a path to a

        Inputs
        ------
        filter:
            A filter from skimage.filters to be applied to the mask
        central_third:
            whether the dilation-erosion should only be computed
            from the most central part of the data

        Output
        ------
        """
        mask_ni = ni.load(mask_path)
        mask_ni = get_cropped_stack_based_on_mask(mask_ni, mask_ni)
        mask = squeeze_dim(mask_ni.get_fdata(), -1)

        if central_third:
            num_z = mask.shape[2]
            center_z = int(num_z / 2.0)
            mask = mask[
                ..., int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]

        assert (
            filter in SKIMAGE_FCT
        ), f"ERROR: {filter} is not a function from `skimage.filters`"

        filtered = filter(mask)
        res = np.mean(abs(filtered - mask))
        return res, np.isnan(res)

    @allow_kwargs
    def _metric_filter(
        self,
        lr_path: str,
        mask_path: str,
        filter=None,
        central_third: bool = True,
    ) -> np.ndarray:
        """Given a path to a LR image and its corresponding image,
        loads and processes the LR image, filters it with a `filter` from
        skimage.filters and returns the mean of the absolute value.

        Inputs
        ------
        filter:
            A filter from skimage.filters to be applied to the mask
        central_third:
            whether the dilation-erosion should only be computed
            from the most central part of the data

        Output
        ------
        """
        im, mask = self._load_and_prep_nifti(
            lr_path, mask_path, crop_image=True, central_third=central_third
        )

        assert (
            filter in SKIMAGE_FCT
        ), f"ERROR: {filter} is not a function from `skimage.filters`"

        filtered = filter(im)
        res = np.mean(abs(filtered - im))
        return res, np.isnan(res)


class SubjectMetrics:
    """TODO"""

    def metric_include_volumes_median(self, sub_volumes_dict):
        """TODO"""
        median_volume = np.median(list(sub_volumes_dict.values()))
        median_volume_dict = {
            k: v > 0.7 * median_volume for k, v in sub_volumes_dict.items()
        }
        median_volume_dict["median"] = median_volume
        return median_volume_dict
