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
import numpy as np
import nibabel as ni
import skimage
import traceback
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
from fetal_brain_qc.utils import squeeze_dim
from scipy.stats import kurtosis, variation
from functools import partial
import pandas as pd
from .mriqc_metrics import (
    summary_stats,
    volume_fraction,
    snr,
    cnr,
    cjv,
    wm2max,
)
import sys

SKIMAGE_FCT = [fct for _, fct in getmembers(skimage.filters, isfunction)]
DEFAULT_METRICS = [
    "centroid",
    "dl_slice_iqa_full",
    "dl_stack_iqa_full",
    "rank_error",
    "mask_volume",
    "ncc",
    "nmi",
]
SEGM = {"BG": 0, "CSF": 1, "GM": 2, "WM": 3}
segm_names = list(SEGM.keys())


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
        verbose=False,
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
        self.verbose = verbose
        if ckpt_stack_iqa:
            import os

            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
            from fetal_brain_qc.fnndsc_IQA import Predictor

            self.stack_predictor = Predictor(ckpt_stack_iqa)
        self.slice_iqa_enabled = True if ckpt_slice_iqa and device else False
        if ckpt_slice_iqa:
            from fetal_brain_qc.fetal_IQA import resnet34
            import torch

            self.device = device
            self.slice_model = resnet34(num_classes=3)
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
            "seg_sstats": self.process_metric(
                self._seg_sstats, type="seg", central_third=True
            ),
            "seg_sstats_full": self.process_metric(
                self._seg_sstats, type="seg", central_third=False
            ),
            "seg_volume": self.process_metric(
                self._seg_volume, type="seg", central_third=True
            ),
            "seg_volume_full": self.process_metric(
                self._seg_volume, type="seg", central_third=False
            ),
            "seg_snr": self.process_metric(
                self._seg_snr, type="seg", central_third=True
            ),
            "seg_snr_full": self.process_metric(
                self._seg_snr, type="seg", central_third=False
            ),
            "seg_cnr": self.process_metric(
                self._seg_cnr, type="seg", central_third=True
            ),
            "seg_cnr_full": self.process_metric(
                self._seg_cnr, type="seg", central_third=False
            ),
            "seg_cjv": self.process_metric(
                self._seg_cjv, type="seg", central_third=True
            ),
            "seg_cjv_full": self.process_metric(
                self._seg_cjv, type="seg", central_third=False
            ),
            "seg_wm2max": self.process_metric(
                self._seg_wm2max, type="seg", central_third=True
            ),
            "seg_wm2max_full": self.process_metric(
                self._seg_wm2max, type="seg", central_third=False
            ),
            "im_size": self._metric_vx_size,
        }

        self._check_metrics()
        self.normalization = None
        self.norm_dict = {}
        # Summary statistics from the segmentation, used for computing a bunch of metrics
        # besides being a metric itself
        self._sstats = None

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
        computes the factor that should be used to scale the input to the LR method.
        """

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

    def get_nan_output(self, metric):
        sstats_keys = [
            "mean",
            "median",
            "median",
            "p95",
            "p05",
            "k",
            "stdv",
            "mad",
            "n",
        ]
        if "seg_" in metric:
            metrics = segm_names
            if "seg_sstats" in metric:
                metrics = [f"{n}_{k}" for n in segm_names for k in sstats_keys]
            return {m: np.nan for m in metrics}
        else:
            return [np.nan]

    def get_default_output(self, metric):
        """Return the default output for a given metric when the mask is invalid and metrics cannot be computed."""
        METRIC_DEFAULT = {"cjv": 0}
        if metric not in METRIC_DEFAULT.keys():
            return [0.0, False]
        else:
            return METRIC_DEFAULT[metric]

    def _flatten_dict(self, d):
        """Flatten a nested dictionary by concatenating the keys with '_'."""
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(
                    {
                        k + "_" + kk: vv
                        for kk, vv in self._flatten_dict(v).items()
                    }
                )
            else:
                out[k] = v
        return out

    def eval_metrics_and_update_results(
        self, results, metric, args_dict, is_valid_mask
    ):
        """Evaluate a metric and update the results dictionary."""
        if is_valid_mask:
            try:
                out = self.metrics_func[metric](**args_dict)
            except Exception:
                if self.verbose:

                    print(
                        f"EXCEPTION with {metric}\n" + traceback.format_exc(),
                        file=sys.stderr,
                    )
                out = self.get_nan_output(metric)
            # Checking once more that if the metric is nan, we replace it with 0
        else:
            out = self.get_default_output(metric)
        if isinstance(out, dict):
            out = self._flatten_dict(out)
            for k, v in out.items():
                results[metric + "_" + k] = v if not np.isnan(v) else 0.0
                results[metric + "_" + k + "_nan"] = np.isnan(v)
        else:
            if np.isnan(out[0]):
                out = (0, True)
            results[metric], results[metric + "_nan"] = out
        return results

    def evaluate_metrics(self, lr_path, mask_path, seg_path=None):
        """Evaluate the metrics for a given LR image and mask.

        Args:
            lr_path (str): Path to the LR image.
            mask_path (str): Path to the mask.
            seg_path (str, optional): Path to the segmentation. Defaults to None.

        Returns:
            dict: Dictionary containing the results of the metrics.
        """

        # Remark: Could do something better with a class here: giving flexible
        # features as input.
        # Reset the summary statistics
        self._sstats = None
        args_dict = {
            "lr_path": lr_path,
            "mask_path": mask_path,
            "seg_path": seg_path,
        }
        if any(["seg_" in m for m in self._metrics]):
            assert seg_path is not None, (
                "Segmentation path should be provided "
                "when evaluating segmentation metrics."
            )
        results = {}
        is_valid_mask = self._valid_mask(mask_path)
        if not is_valid_mask:
            print(f"\tWARNING: Empty mask {mask_path}.")
        for m in self._metrics:
            if self.verbose:
                print("\tRunning", m)
            results = self.eval_metrics_and_update_results(
                results, m, args_dict, is_valid_mask
            )
        return results

    def _check_metrics(self):
        """Check that the metrics are valid."""

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
        """Wrapper to process the different categories of metrics.

        Args:
            metric (str): Name of the metric (in the list of available metrics).
            type (str, optional): Type of metric. Defaults to "ref". Available types are:
                - "ref": metric that is computed by comparing neighbouring slices.
                - "noref": metric that relies on individual slices
                - "dl": metric that leverage a DL model.
                - "seg": metric that make use of a segmentation.
            **kwargs: Additional processing to be done before evaluating the metric, detailed in the docstring of the corresponding function.
        """

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
        elif type == "seg":
            return freeze(
                self.preprocess_and_evaluate_seg_metric,
                seg_metric=metric,
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
            lr_path (str): Path to the low-resolution stack
            mask_path (str): Path to the mask
            threshold (float, optional): Threshold for the svd_error. Defaults to 0.99.
            central_third (bool, optional): Whether to only consider the central third of the stack. Defaults to True.
            crop_image (bool, optional): Whether to crop the image based on the mask. Defaults to True.
            relative_rank (bool, optional): Whether to use the relative rank (rank/num_slices) or the absolute rank. Defaults to True.

        Returns:
            rank_error (float): The computed rank_error (rank * svd_error)
            isnan (bool): Whether the computed value is nan

        """
        image_ni = ni.load(lr_path)
        image = image_ni.get_fdata()
        mask_ni = ni.load(mask_path)

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

    def load_and_format_seg(self, seg_path):
        """Load segmentation and format it to be used by the metrics"""

        seg_path = str(seg_path).strip()
        if seg_path.endswith(".nii.gz"):
            seg_ni = ni.load(seg_path)
            seg = squeeze_dim(seg_ni.get_fdata(), -1).transpose(2, 1, 0)
            if seg.max() > 3:
                seg[seg == 4] = 1
                seg[seg == 6] = 2
                seg[seg > 3] = 0
            seg = seg.transpose(2, 1, 0)
            seg_dict = {
                k: (seg == l).astype(np.uint8) for k, l in SEGM.items()
            }

        elif seg_path.endswith(".npz"):
            seg = np.load(seg_path)["probabilities"]
            if seg.shape[0] > 4:
                seg[1] += seg[4]
                seg[2] += seg[6]
                seg = seg[:4]

            seg = seg.transpose(0, 3, 2, 1)
            seg_dict = {k: seg[l] for k, l in SEGM.items()}
        else:
            raise ValueError(
                f"Unknown file format for segmentation file {seg_path}"
            )
        # We cannot return a nifti object as seg_path might be .npz
        return seg_dict

    def _load_and_prep_nifti(
        self,
        lr_path,
        mask_path,
        seg_path=None,
        *,
        crop_image=True,
        central_third=True,
    ):
        """Load and prepare the nifti files for the metrics computation.

        Args:
            lr_path (str): Path to the low-resolution stack
            mask_path (str): Path to the mask
            seg_path (str, optional): Path to the segmentation. Defaults to None.
            crop_image (bool, optional): Whether to crop the image based on the mask. Defaults to True.
            central_third (bool, optional): Whether to only consider the central third of the stack. Defaults to True.

        Returns:
            image (np.ndarray): The low-resolution stack
            mask (np.ndarray): The mask
            seg (np.ndarray): The segmentation
        """

        image_ni = ni.load(lr_path)
        image = squeeze_dim(image_ni.get_fdata(), -1).transpose(2, 1, 0)
        mask_ni = ni.load(mask_path)
        mask = squeeze_dim(mask_ni.get_fdata(), -1).transpose(2, 1, 0)
        if seg_path is not None and not crop_image:
            raise ValueError(
                "If seg_path is not None, crop_image must be True"
            )
        if mask.sum() == 0.0:
            return None, None

        if crop_image:
            # Recent change: put a margin of 15mm around the image in for segmentation.
            # This is needed for segmentation.
            margin = 15 if seg_path is not None else 0

            def crop_stack(x, y):
                return get_cropped_stack_based_on_mask(
                    x,
                    y,
                    boundary_i=margin,
                    boundary_j=margin,
                    boundary_k=margin,
                )

            imagec = crop_stack(image_ni, mask_ni)
            maskc = crop_stack(mask_ni, mask_ni)
            if image is None or mask is None:
                return None, None
            image = squeeze_dim(imagec.get_fdata(), -1).transpose(2, 1, 0)
            mask = squeeze_dim(maskc.get_fdata(), -1).transpose(2, 1, 0)
            if seg_path is not None:
                # The segmentation map is computed on data cropped with margin 15mm
                seg_dict = {
                    k: crop_stack(
                        ni.Nifti1Image(v, imagec.affine, imagec.header),
                        maskc,
                    )
                    for k, v in self.load_and_format_seg(seg_path).items()
                }

                seg_dict = {
                    k: squeeze_dim(v.get_fdata(), -1).transpose(2, 1, 0)
                    for k, v in seg_dict.items()
                }

                assert image.shape == (
                    seg_dict["BG"].shape
                ), "Image and segmentation have different sizes"
        if central_third:
            num_z = image.shape[0]
            center_z = int(num_z / 2.0)
            image = image[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0) + 1
            ]
            mask = mask[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0) + 1
            ]
            if seg_path is not None:
                seg_dict = {
                    k: v[
                        int(center_z - num_z / 6.0) : int(
                            center_z + num_z / 6.0
                        )
                        + 1
                    ]
                    for k, v in seg_dict.items()
                }

        if self.normalization is not None:
            norm = self.norm_dict[lr_path]
            image = image * norm
        if seg_path is not None:
            return image, mask, seg_dict
        else:
            return image, mask

    def _ssim(
        self,
        lr_path,
        mask_path,
        seg_path=None,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
        use_window=False,
        window_size=3,
    ):
        image, mask = self._load_and_prep_nifti(
            lr_path,
            mask_path,
            crop_image=crop_image,
            central_third=central_third,
        )

        if (
            image is None
            or mask is None
            or any([s < 7 for s in image.shape[1:]])
        ):
            # image is None when the mask is empty: nothing is computed.
            # Similarly, return None when the image is of size smaller than 7
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

    def _seg_sstats(self, image, segmentation):
        self._sstats = summary_stats(image, segmentation)
        return self._sstats

    def _seg_volume(self, image, segmentation):
        return volume_fraction(segmentation)

    def _seg_snr(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        snr_dict = {}
        for tlabel in segmentation.keys():
            snr_dict[tlabel] = snr(
                self._sstats[tlabel]["median"],
                self._sstats[tlabel]["stdv"],
                self._sstats[tlabel]["n"],
            )
        snr_dict["total"] = float(np.mean(list(snr_dict.values())))
        return snr_dict

    def _seg_cnr(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = cnr(
            self._sstats["WM"]["median"],
            self._sstats["GM"]["median"],
            self._sstats["BG"]["stdv"],
            self._sstats["WM"]["stdv"],
            self._sstats["GM"]["stdv"],
        )
        is_nan = np.isnan(out)
        return 0.0 if is_nan else out, is_nan

    def _seg_cjv(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = cjv(
            # mu_wm, mu_gm, sigma_wm, sigma_gm
            self._sstats["WM"]["median"],
            self._sstats["GM"]["median"],
            self._sstats["WM"]["mad"],
            self._sstats["GM"]["mad"],
        )
        is_nan = np.isnan(out)
        return 1000 if is_nan else out, is_nan

    def _seg_wm2max(self, image, segmentation):
        if self._sstats is None:
            self._sstats = summary_stats(image, segmentation)
        out = wm2max(image, self._sstats["WM"]["median"])
        is_nan = np.isnan(out)
        return 0.0 if is_nan else out, is_nan

    def preprocess_and_evaluate_metric(
        self,
        metric,
        lr_path,
        mask_path,
        seg_path=None,
        *,
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
            lr_path,
            mask_path,
            crop_image=crop_image,
            central_third=central_third,
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
        seg_path=None,
        *,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        flatten=True,
    ):
        image, mask = self._load_and_prep_nifti(
            lr_path,
            mask_path,
            crop_image=crop_image,
            central_third=central_third,
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
        seg_path=None,
        *,
        central_third=True,
        crop_image=True,
        positive_only=False,
    ):
        image, mask = self._load_and_prep_nifti(
            lr_path,
            mask_path,
            crop_image=crop_image,
            central_third=central_third,
        )

        metric = dl_metric(image, mask, positive_only=False)
        return metric, np.isnan(metric)

    def preprocess_and_evaluate_seg_metric(
        self,
        seg_metric,
        lr_path,
        mask_path,
        seg_path,
        *,
        central_third=True,
        crop_image=True,
    ):
        assert (
            seg_path is not None
        ), "Segmentation path must be provided for seg metrics."
        image, _, seg = self._load_and_prep_nifti(
            lr_path,
            mask_path,
            seg_path,
            crop_image=crop_image,
            central_third=central_third,
        )
        return seg_metric(image, seg)

    @allow_kwargs
    def _metric_stack_iqa(self, image, mask, positive_only=None) -> np.ndarray:
        """ """
        from fetal_brain_qc.fnndsc_IQA import fnndsc_preprocess

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

    def _metric_vx_size(
        self, lr_path: str, mask_path: str, seg_path: str = None
    ):
        """Given a path to a LR image and its corresponding image,
        loads the LR image and return the voxel size.
        """
        im = ni.load(lr_path)

        x, y, z = im.header.get_zooms()
        out_dict = {
            "x": x,
            "y": y,
            "z": z,
            "vx_size": x * y * z,
            "ip_size": x * y,
        }
        return out_dict


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
