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
)
from fetal_brain_qc.utils import get_cropped_stack_based_on_mask
from fetal_brain_qc.fnndsc_IQA import fnndsc_preprocess

DEFAULT_METRICS = [
    "dl_slice_iqa",
    "dl_stack_iqa",
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
            "ncc": freeze(
                self._metric_normalized_cross_correlation, **default_params
            ),
            "ncc_median": freeze(
                self._metric_normalized_cross_correlation,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "ncc_intersection": freeze(
                self._metric_normalized_cross_correlation,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "ncc_full": freeze(
                self._metric_normalized_cross_correlation,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "joint_entropy": freeze(
                self._metric_joint_entropy, **default_params
            ),
            "joint_entropy_median": freeze(
                self._metric_joint_entropy,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "joint_entropy_intersection": freeze(
                self._metric_joint_entropy,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "joint_entropy_full": freeze(
                self._metric_joint_entropy,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "mi": freeze(self._metric_mutual_information, **default_params),
            "mi_median": freeze(
                self._metric_mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "mi_intersection": freeze(
                self._metric_mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "mi_full": freeze(
                self._metric_mutual_information,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "nmi": freeze(
                self._metric_normalized_mutual_information, **default_params
            ),
            "nmi_median": freeze(
                self._metric_normalized_mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "nmi_intersection": freeze(
                self._metric_normalized_mutual_information,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "nmi_full": freeze(
                self._metric_normalized_mutual_information,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "shannon_entropy": freeze(
                self._metric_joint_entropy, **default_params
            ),
            "shannon_entropy_median": freeze(
                self._metric_joint_entropy,
                central_third=True,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=False,
                reduction="median",
            ),
            "shannon_entropy_full": freeze(
                self._metric_joint_entropy,
                central_third=False,
                crop_image=True,
                compute_on_mask=True,
                mask_intersection=True,
                reduction="mean",
            ),
            "dl_stack_iqa": self._metric_stack_iqa,
            "dl_slice_iqa": self._metric_slice_iqa,
        }

        self._check_metrics()

    def get_default_metrics(self):
        return DEFAULT_METRICS

    def get_all_metrics(self):
        return list(self.metrics_func.keys())

    def _valid_mask(self, mask_path):
        mask = ni.load(mask_path).get_fdata()
        if mask.sum() == 0:
            return False
        else:
            return True

    def evaluate_metrics(self, lr_path, mask_path):
        """TODO"""
        # Remark: Could do something better with a class here: giving flexible
        # features as input.
        args_dict = {"lr_path": lr_path, "mask_path": mask_path}
        results = {}
        if not self._valid_mask(mask_path):
            print(f"\tWARNING: Empty mask {mask_path}.")
        for m in self._metrics:
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

    @classmethod
    @allow_kwargs
    def _metric_mask_centroid(
        cls, mask_path: str, central_third: bool = True
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
        mask = mask_ni.get_fdata().squeeze()
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

    @classmethod
    @allow_kwargs
    def _metric_mask_volume(cls, mask_path):
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
        mask = mask_ni.get_fdata().squeeze()
        vx_volume = np.array(mask_ni.header.get_zooms()).prod()
        isnan = False
        return np.sum(mask) * vx_volume, isnan

    @classmethod
    @allow_kwargs
    def _metric_rank_error(
        cls,
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
        mask = mask_ni.get_fdata().squeeze()

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

    @classmethod
    def _load_and_prep_nifti(
        cls, lr_path, mask_path, crop_image, central_third
    ):
        """TODO"""
        image_ni = ni.load(lr_path)
        image = image_ni.get_fdata().transpose(2, 1, 0)
        mask_ni = ni.load(mask_path)
        mask = mask_ni.get_fdata().squeeze().transpose(2, 1, 0)

        if crop_image:
            image = get_cropped_stack_based_on_mask(image_ni, mask_ni)
            mask = get_cropped_stack_based_on_mask(mask_ni, mask_ni)
            if image is None or mask is None:
                return None, None

            image = image.get_fdata().transpose(2, 1, 0)
            mask = mask.get_fdata().transpose(2, 1, 0)
        if central_third:
            num_z = image.shape[0]
            center_z = int(num_z / 2.0)
            image = image[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]
            mask = mask[
                int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]
        return image, mask

    @classmethod
    def preprocess_and_evaluate_metric(
        cls,
        metric,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
    ):

        VALID_REDUCTIONS = ["mean", "median"]
        assert reduction in VALID_REDUCTIONS, (
            f"Unknown reduction function {reduction}."
            f"Choose from {VALID_REDUCTIONS}"
        )
        image, mask = cls._load_and_prep_nifti(
            lr_path, mask_path, crop_image, central_third
        )
        if image is None or mask is None:
            # image is None when the mask is empty: nothing is computed.
            return np.nan, True
        metric_array = np.zeros((image.shape[0], image.shape[0]))
        # TODO: This computation is run between all slices in an image.
        # What about doing something on a window around a given image?
        for i, img_i in enumerate(image):
            for j, img_j in enumerate(image):
                if compute_on_mask:
                    idx = (
                        np.where(mask[i] * mask[j])
                        if mask_intersection
                        else np.where(mask[i] + mask[j])
                    )
                    metric_array[i, j] = metric(img_i[idx], img_j[idx])
                else:

                    metric_array[i, j] = metric(img_i, img_j)

        if reduction == "mean":
            return np.nanmean(metric_array), np.isnan(metric_array).any()
        elif reduction == "median":
            return np.nanmedian(metric_array), np.isnan(metric_array).any()

    @classmethod
    def preprocess_and_evaluate_noref_metric(
        cls,
        noref_metric,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
    ):
        image, mask = cls._load_and_prep_nifti(
            lr_path, mask_path, crop_image, central_third
        )

        if compute_on_mask:
            image = image[np.where(mask)]
        return noref_metric(image.flatten())

    @classmethod
    @allow_kwargs
    def _metric_normalized_cross_correlation(
        cls,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
    ) -> np.ndarray:
        """TODO
        Computes the normalized cross correlation on a given stack at lr_path

        Inputs
        ------
        lr_path:
        mask_path:
        central_third:
        compute_on_mask:
        mask_intersection:
        reduction:

        Output
        ------
        The normalized cross correlation on the stack as a scalar in the range -1, 1
        """
        return cls.preprocess_and_evaluate_metric(
            normalized_cross_correlation,
            lr_path,
            mask_path,
            central_third,
            crop_image,
            compute_on_mask,
            mask_intersection,
            reduction,
        )

    @classmethod
    @allow_kwargs
    def _metric_shannon_entropy(
        cls,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
    ) -> np.ndarray:
        """TODO
        Computes the Shannon entropy on a given stack at lr_path

        Inputs
        ------
        lr_path:
        mask_path:
        central_third:
        compute_on_mask:

        Output
        ------
        The joint entropy on the stack as scalar value in [0, log_b(n)]
        """
        return cls.preprocess_and_evaluate_noref_metric(
            shannon_entropy,
            lr_path,
            mask_path,
            central_third,
            crop_image,
            compute_on_mask,
        )

    @classmethod
    @allow_kwargs
    def _metric_joint_entropy(
        cls,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
    ) -> np.ndarray:
        """TODO
        Computes the joint entropy on a given stack at lr_path

        Inputs
        ------
        lr_path:
        mask_path:
        central_third:
        compute_on_mask:
        mask_intersection:
        reduction:

        Output
        ------
        The joint entropy on the stack as scalar value >= 0
        """
        return cls.preprocess_and_evaluate_metric(
            joint_entropy,
            lr_path,
            mask_path,
            central_third,
            crop_image,
            compute_on_mask,
            mask_intersection,
            reduction,
        )

    @classmethod
    @allow_kwargs
    def _metric_mutual_information(
        cls,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
    ) -> np.ndarray:
        """TODO
        Computes the mutual information on a given stack at lr_path

        Inputs
        ------
        lr_path:
        mask_path:
        central_third:
        compute_on_mask:
        mask_intersection:
        reduction:

        Output
        ------
        The normalized mutual information on the stack as a scalar >= 0
        """
        return cls.preprocess_and_evaluate_metric(
            mutual_information,
            lr_path,
            mask_path,
            central_third,
            crop_image,
            compute_on_mask,
            mask_intersection,
            reduction,
        )

    @classmethod
    @allow_kwargs
    def _metric_normalized_mutual_information(
        cls,
        lr_path,
        mask_path,
        central_third=True,
        crop_image=True,
        compute_on_mask=True,
        mask_intersection=True,
        reduction="mean",
    ) -> np.ndarray:
        """TODO
        Computes the normalized mutual information on a given stack at lr_path

        Inputs
        ------
        lr_path:
        mask_path:
        central_third:
        compute_on_mask:
        mask_intersection:
        reduction:

        Output
        ------
        The normalized mutual information on the stack as scalar value >= 0
        with upper bound as in Skouson2001

        Skouson, M.B., Quji Guo & Zhi-Pei Liang, 2001. A bound on mutual
             information for image registration. IEEE Transactions on Medical
             Imaging, 20(8), pp.843-846.
        """
        return cls.preprocess_and_evaluate_metric(
            normalized_mutual_information,
            lr_path,
            mask_path,
            central_third,
            crop_image,
            compute_on_mask,
            mask_intersection,
            reduction,
        )

    @allow_kwargs
    def _metric_stack_iqa(
        self,
        lr_path,
        mask_path,
    ) -> np.ndarray:
        """ """
        # Loading data

        img = ni.load(lr_path).get_fdata()
        mask = ni.load(mask_path).get_fdata().squeeze()

        img = fnndsc_preprocess(img, mask)
        df = self.stack_predictor.predict([img], ["img"])
        df = df.set_index("filename")
        return df.loc["img"]["quality"], False

    @allow_kwargs
    def _metric_slice_iqa(
        self,
        lr_path,
        mask_path,
    ) -> np.ndarray:
        """ """
        # Loading data
        from fetal_brain_qc.fetal_IQA import eval_model

        iqa_dict = eval_model(
            lr_path, mask_path, self.slice_model, self.device
        )
        weighted_score = iqa_dict[list(iqa_dict.keys())[0]]["weighted"]
        p_good, p_bad = [], []
        for v in iqa_dict.values():
            p_good.append(v["good"])
            p_bad.append(v["bad"])
        weighted_score = (sum(p_good) - sum(p_bad)) / len(p_good)
        return weighted_score, False


class SubjectMetrics:
    """TODO"""

    @classmethod
    def metric_include_volumes_median(cls, sub_volumes_dict):
        """TODO"""
        median_volume = np.median(list(sub_volumes_dict.values()))
        median_volume_dict = {
            k: v > 0.7 * median_volume for k, v in sub_volumes_dict.items()
        }
        median_volume_dict["median"] = median_volume
        return median_volume_dict
