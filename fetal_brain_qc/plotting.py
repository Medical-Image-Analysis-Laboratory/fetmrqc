import nibabel as ni
import numpy as np
import os
from pathlib import Path
import os.path as op
import matplotlib.pyplot as plt
from mialsrtk_utils import get_cropped_stack_based_on_mask
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

    est_vmin, est_vmax = _get_limits(dslice)
    if not vmin:
        vmin = est_vmin
    if not vmax:
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
        vmin = np.percentile(data[data_mask], 0.5)
        vmax = np.percentile(data[data_mask], 99.5)

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
    maskc = get_cropped_stack_based_on_mask(
        mask,
        mask,
        boundary_i=boundary,
        boundary_j=boundary,
        boundary_k=boundary_tp,
    )
    zooms = im.header.get_zooms()

    im_data = imc.get_fdata()
    # Remove extra dimensions
    im_data = np.squeeze(im_data)

    n_slices = im_data.shape[2]
    mid_img = im_data.shape[0] // 2
    nslices = im_data.shape[2]

    nrows = math.ceil(nslices / ncols_ip)

    fig = plt.figure(figsize=(12, nrows * 2))

    vmin, vmax = _get_limits(im_data, only_plot_noise=False)

    naxis = 1
    for z_val in range(nslices):
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
                [mask.get_fdata()[:, i, :].sum() for i in range(mask.shape[1])]
            )
        )[0].mean()
    )

    min_x = max(mid_x - n_slices_tp // 2 * every_n_tp, 0)
    max_x = min(
        mid_x + n_slices_tp // 2 * every_n_tp - every_n_tp // 2, mask.shape[1]
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
                [mask.get_fdata()[i, :, :].sum() for i in range(mask.shape[0])]
            )
        )[0].mean()
    )

    min_y = max(mid_y - n_slices_tp // 2 * every_n_tp, 0)
    max_y = min(
        mid_y + n_slices_tp // 2 * every_n_tp - every_n_tp // 2, mask.shape[0]
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
        f"{report_dir}/thp1_mosaic.svg",
        f"{report_dir}/thp2_mosaic.svg",
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
