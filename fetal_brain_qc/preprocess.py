import os
import SimpleITK as sitk
import nibabel as ni
from .utils import get_cropped_stack_based_on_mask


def crop_input(file_path, mask_path, mask_image, dir_cropped):
    """Crops input image and mask, optionally masks it and
    saves the files to dir_cropped.
    """
    os.makedirs(dir_cropped, exist_ok=True)
    output = os.path.join(dir_cropped, os.path.basename(file_path))
    output_mask = os.path.join(dir_cropped, os.path.basename(mask_path))

    im, m = ni.load(file_path), ni.load(mask_path)
    boundary_mm = 15
    imc = get_cropped_stack_based_on_mask(
        im,
        m,
        boundary_i=boundary_mm,
        boundary_j=boundary_mm,
        boundary_k=boundary_mm,
    )

    maskc = get_cropped_stack_based_on_mask(
        m,
        m,
        boundary_i=boundary_mm,
        boundary_j=boundary_mm,
        boundary_k=boundary_mm,
    )

    if mask_image:
        imc = ni.Nifti1Image(imc.get_fdata() * maskc.get_fdata(), imc.affine)
    else:  # Masking
        imc = ni.Nifti1Image(imc.get_fdata(), imc.affine)

    ni.save(imc, output)
    ni.save(maskc, output_mask)
    return output, output_mask


def correct_bias_field(
    file_path,
    mask_path,
    dir_output,
    use_mask,
    bias_field_fwhm,
    convergence_threshold,
    spline_order,
    wiener_filter_noise,
):
    """Bias field correction using sitk."""

    bias_field_corrector = sitk.N4BiasFieldCorrectionImageFilter()

    bias_field_corrector.SetBiasFieldFullWidthAtHalfMaximum(bias_field_fwhm)
    bias_field_corrector.SetConvergenceThreshold(convergence_threshold)
    bias_field_corrector.SetSplineOrder(spline_order)
    bias_field_corrector.SetWienerFilterNoise(wiener_filter_noise)

    # Extract the base path from a .nii.gz file
    bias_path = file_path[:-7] + "_bias" + file_path[-7:]
    # Output files
    output = os.path.join(dir_output, os.path.basename(file_path))
    output_bias = os.path.join(dir_output, os.path.basename(bias_path))

    image_sitk = sitk.ReadImage(str(file_path), sitk.sitkFloat64)
    if use_mask:
        output_mask = os.path.join(dir_output, os.path.basename(mask_path))
        sitk_mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        sitk_mask.CopyInformation(image_sitk)
        bias_corr = bias_field_corrector.Execute(image_sitk, sitk_mask)

        stack_corrected_sitk_mask = sitk.Resample(
            sitk_mask,
            image_sitk,
            sitk.Euler3DTransform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk_mask.GetPixelIDValue(),
        )
        sitk.WriteImage(stack_corrected_sitk_mask, output_mask)
    else:
        bias_corr = bias_field_corrector.Execute(image_sitk)

    sitk.WriteImage(bias_corr, output)

    bias = bias_field_corrector.GetLogBiasFieldAsImage(image_sitk)
    sitk.WriteImage(bias, output_bias)
