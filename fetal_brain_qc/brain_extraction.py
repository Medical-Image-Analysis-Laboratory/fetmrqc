import os
from fetal_brain_utils import iter_bids
from .utils import fill_pattern
from pathlib import Path
import shutil


def bidsify_monaifbs(bids_layout, mask_pattern, masks_tmp):
    """Given masks generated using MONAIfbs, saved in `masks_tmp`,
    moves the saved masks to their BIDS location specified in `mask_pattern`.
    """

    # Construct the MONAIfbs pattern
    monaifbs_pattern = os.path.join(
        os.path.abspath(masks_tmp),
        "sub-{subject}[_ses-{session}][_acq-{acquisition}][_run-{run}]_T2w/"
        "sub-{subject}[_ses-{session}][_acq-{acquisition}][_run-{run}]_T2w_mask.nii.gz",
    )
    for sub, ses, run, _ in iter_bids(bids_layout):
        # Generate the MONAIfbs pattern
        monaifbs_mask_path = fill_pattern(
            bids_layout, sub, ses, run, monaifbs_pattern
        )
        # Generate the target BIDS pattern
        mask_path = Path(
            fill_pattern(bids_layout, sub, ses, run, mask_pattern)
        )
        os.makedirs(mask_path.parent, exist_ok=True)
        shutil.move(monaifbs_mask_path, mask_path)
    shutil.rmtree(os.path.abspath(masks_tmp))


def run_brain_extraction(in_files, out_dir, brain_ckpt):
    """Running the brain extraction using MONAIfbs.
    The original code is from https://github.com/gift-surg/MONAIfbs/tree/main
    """
    import monaifbs
    from monaifbs.src.inference.monai_dynunet_inference import run_inference
    import yaml

    config_file = os.path.join(
        *[
            os.path.dirname(monaifbs.__file__),
            "config",
            "monai_dynUnet_inference_config.yml",
        ]
    )
    if not os.path.isfile(brain_ckpt):
        raise ValueError(f"MONAIfbs ckpt not found at {brain_ckpt}.")

    with open(config_file) as f:
        print("*** Config file")
        print(config_file)
        config = yaml.load(f, Loader=yaml.FullLoader)

    # add the output directory to the config dictionary
    config["output"] = {
        "out_postfix": "mask",
        "out_dir": out_dir,
    }
    os.makedirs(config["output"]["out_dir"], exist_ok=True)
    config["inference"]["model_to_load"] = brain_ckpt

    run_inference(in_files, config)
