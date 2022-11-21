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
