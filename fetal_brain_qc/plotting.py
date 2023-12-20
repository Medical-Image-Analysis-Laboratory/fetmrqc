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
import nibabel as ni
import numpy as np
import os
from pathlib import Path
import os.path as op
import matplotlib.pyplot as plt
from fetal_brain_utils import get_cropped_stack_based_on_mask
import math


def plot_slice(
    dslice,
    spacing=None,
    cmap="Greys_r",
    label=None,
    ax=None,
    vmax=None,
    vmin=None,
    annotate=False,
):
    """From MRIQC"""
    from matplotlib.cm import get_cmap

    if isinstance(cmap, (str, bytes)):
        cmap = get_cmap(cmap)

    if vmin is None or vmax is None:
        est_vmin, est_vmax = _get_limits(dslice)
    if vmin is None:
        vmin = est_vmin
    if vmax is None:
        vmax = est_vmax

    if ax is None:
        ax = plt.gca()

    if spacing is None:
        spacing = [1.0, 1.0]

    phys_sp = np.array(spacing) * dslice.shape
    ax.imshow(
        np.swapaxes(dslice, 0, 1),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        extent=[0, phys_sp[0], 0, phys_sp[1]],
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.axis("off")

    bgcolor = cmap(min(vmin, 0.0))
    fgcolor = cmap(vmax)

    if annotate:
        ax.text(
            0.95,
            0.95,
            "R",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )
        ax.text(
            0.05,
            0.95,
            "L",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    if label is not None:
        ax.text(
            0.98,
            0.01,
            label,
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    return ax


def _get_limits(nifti_file, only_plot_noise=False):
    """From MRIQC"""
    if isinstance(nifti_file, str):
        nii = ni.as_closest_canonical(ni.load(nifti_file))
        data = nii.get_fdata()
    else:
        data = nifti_file

    data_mask = np.logical_not(np.isnan(data))

    if only_plot_noise:
        data_mask = np.logical_and(data_mask, data != 0)
        vmin = np.percentile(data[data_mask], 0)
        vmax = np.percentile(data[data_mask], 61)
    else:
        vmin = np.percentile(data[data_mask], 0.01)
        vmax = np.percentile(data[data_mask], 99.9)

    return vmin, vmax


def plot_mosaic(
    imp,
    maskp,
    boundary=20,
    boundary_tp=10,
    ncols_ip=6,
    n_slices_tp=6,
    every_n_tp=4,
    annotate=False,
    cmap="Greys_r",
    report_dir="tmp_report",
):
    """Inspired from MRIQC.
    imp:
        Path to the brain LR T2w image to be plotted
    maskp:
        Path to the brain mask associated with imp
    boundary:
        Boundary to be left around the image when cropping it.
    ncols_ip:
        Number of columns in the in-plane plot
    nslices_tp:
        Number of slices to be displayed in the through-plane views.
    every_n_tp:
        Separation between two slices in the through-plane views.
    annotate:
        Whether the plots should be annotated
    cmap:
        Colormap to be used
    """
    im = ni.load(imp)
    mask = ni.load(maskp)
    imc = get_cropped_stack_based_on_mask(
        im,
        mask,
        boundary_i=boundary,
        boundary_j=boundary,
        boundary_k=boundary_tp,
    )

    zooms = im.header.get_zooms()

    im_data = imc.get_fdata()

    nslices = im_data.shape[2]

    nrows = math.ceil(nslices / ncols_ip)

    fig = plt.figure(figsize=(12, nrows * 2))

    vmin, vmax = _get_limits(im_data, only_plot_noise=False)

    naxis = 1
    for z_val in range(nslices - 1, -1, -1):
        ax = fig.add_subplot(nrows, ncols_ip, naxis)
        plot_slice(
            im_data[:, :, z_val],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            spacing=zooms[:2],
            label="%d" % z_val,
            annotate=annotate,
        )

        naxis += 1

    nrows = math.ceil(n_slices_tp / 2)
    mid_x = int(
        np.nonzero(
            np.array(
                [mask.get_fdata()[i, :, :].sum() for i in range(mask.shape[0])]
            )
        )[0].mean()
    )

    min_x = max(mid_x - n_slices_tp // 2 * every_n_tp, 0)
    max_x = min(
        mid_x + n_slices_tp // 2 * every_n_tp - every_n_tp // 2, mask.shape[0]
    )

    fig2 = plt.figure(figsize=(12, math.ceil(nrows * 4 / 3)))

    naxis = 1

    for x_val in range(min_x, max_x, every_n_tp):
        ax = fig2.add_subplot(nrows, 2, naxis)

        plot_slice(
            im.get_fdata()[x_val, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            spacing=zooms[1:],
            label="%d" % x_val,
            annotate=annotate,
        )
        naxis += 1

    mid_y = int(
        np.nonzero(
            np.array(
                [mask.get_fdata()[:, i, :].sum() for i in range(mask.shape[1])]
            )
        )[0].mean()
    )

    min_y = max(mid_y - n_slices_tp // 2 * every_n_tp, 0)
    max_y = min(
        mid_y + n_slices_tp // 2 * every_n_tp - every_n_tp // 2, mask.shape[1]
    )

    naxis = 1
    fig3 = plt.figure(figsize=(12, math.ceil(nrows * 4 / 3)))
    for y_val in range(min_y, max_y, every_n_tp):
        ax = fig3.add_subplot(3, 2, naxis)

        plot_slice(
            im.get_fdata()[:, y_val, :],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            spacing=zooms[1:],
            label="%d" % y_val,
            annotate=annotate,
        )
        naxis += 1

    os.makedirs(report_dir, exist_ok=True)

    out_files = [
        f"{report_dir}/ip_mosaic.svg",
        f"{report_dir}/tp1_mosaic.svg",
        f"{report_dir}/tp2_mosaic.svg",
    ]

    for f, out in zip([fig, fig2, fig3], out_files):
        f.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.05,
            hspace=0.05,
        )

        f.subplots_adjust(wspace=0.002, hspace=0.002)

        f.savefig(out, format="svg", dpi=300, bbox_inches="tight")
        plt.close(f)
    out_files = [op.abspath(fname) for fname in out_files]
    return out_files


def plot_mosaic_sr(
    imp,
    maskp,
    boundary=0,
    ncols=6,
    annotate=False,
    cmap="Greys_r",
    report_dir="tmp_report",
):
    """Inspired from MRIQC.
    imp:
        Path to the brain LR T2w image to be plotted
    boundary:
        Boundary to be left around the image when cropping it.
    ncols:
        Number of columns
    annotate:
        Whether the plots should be annotated
    cmap:
        Colormap to be used
    """
    im = ni.load(imp)
    if maskp == "":
        mask = ni.load(imp)
        mask = ni.Nifti1Image(
            (mask.get_fdata() > 0).astype(int), mask.affine, mask.header
        )
    else:
        mask = ni.load(maskp)
    imc = get_cropped_stack_based_on_mask(
        im,
        mask,
        boundary_i=boundary,
        boundary_j=boundary,
        boundary_k=boundary,
    )

    zooms = im.header.get_zooms()

    im_data = imc.get_fdata()

    vmin, vmax = _get_limits(im_data, only_plot_noise=False)

    # Use the affine to re-order the axes in standardized manner for visualization
    # 1. Extract the affine orientation (non-zero entry in the line)
    affine_axes = tuple(np.nonzero(im.affine[:3, :3]))
    # Define axes to be flipped based on whether the entry is positive.
    flip_axis = np.nonzero(np.sign(im.affine[affine_axes]).astype(int) == -1)[
        0
    ]
    affine_axes = affine_axes[1]
    # Flip axes
    im_data = np.flip(im_data, axis=flip_axis)

    # This is mysterious to me. This is needed to that I have the axial plane in the Right-Left direction
    # This is only needed for NeSVoR also, not for NiftyMIC. This isn't clear to me why.
    # im_data = im_data[:, :, ::-1]
    # Swap axes
    im_data = im_data.transpose(affine_axes)

    def plot_axis(im, axis, vmin, vmax, cmap, zooms, annotate, reverse=False):
        """Plot a 3D volume along a given axis."""
        axes = [0, 1, 2]
        axes.remove(axis)
        axes = tuple(axes)
        # Generate the mask from the image.
        mask = (im > 0).astype(int)
        plot_sum = mask.sum(axis=axes)
        # Define the range of slices to be plotted and exclude
        # slices with too little pixels of interest.
        plot_range = np.nonzero(
            mask.sum(axis=axes) > (np.median(plot_sum) * 0.2)
        )[0]

        nrows = math.ceil(im.shape[axis] / ncols)

        min_, max_ = min(plot_range), max(plot_range)

        if not reverse:
            iter_range = range(min_, max_, 1)
        else:
            iter_range = range(max_ - 1, min_ - 1, -1)

        im = np.moveaxis(im, axis, 0)
        spacing = zooms[axis]
        fig = plt.figure(figsize=(12, nrows * 2))

        naxis = 1
        for i in iter_range:
            ax = fig.add_subplot(nrows, ncols, naxis)
            plot_slice(
                im[i, :, :],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                spacing=spacing,
                label="%d" % i,
                annotate=annotate,
            )

            naxis += 1
        return fig

    fig = plot_axis(im_data, 2, vmin, vmax, cmap, zooms, annotate)
    # I don't know why we have to swap the axes on the sagittal axis ...
    # This is mysterious to me. This is needed to that I have the sagittal plane in the Anterior-Posterior direction
    fig2 = plot_axis(im_data[:, ::-1, :], 0, vmin, vmax, cmap, zooms, annotate)
    fig3 = plot_axis(im_data, 1, vmin, vmax, cmap, zooms, annotate)

    os.makedirs(report_dir, exist_ok=True)

    out_files = [
        f"{report_dir}/axial_mosaic.svg",
        f"{report_dir}/sagittal_mosaic.svg",
        f"{report_dir}/coronal_mosaic.svg",
    ]

    for f, out in zip([fig, fig2, fig3], out_files):
        f.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.05,
            hspace=0.05,
        )

        f.subplots_adjust(wspace=0.002, hspace=0.002)

        f.savefig(out, format="svg", dpi=300, bbox_inches="tight")
        plt.close(f)
    out_files = [op.abspath(fname) for fname in out_files]
    return out_files
