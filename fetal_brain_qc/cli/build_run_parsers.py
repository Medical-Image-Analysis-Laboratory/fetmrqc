import argparse
from fetal_brain_qc.definitions import MASK_PATTERN, BRAIN_CKPT


def build_inference_parser(parser):
    parser.add_argument(
        "--bids_dir",
        required=True,
        help="BIDS directory containing the LR series.",
    )

    parser.add_argument(
        "--masks_dir",
        help=(
            "Root of the BIDS directory where brain masks will be/are stored. "
            "If masks already exist, they will be used."
        ),
        required=True,
    )

    parser.add_argument(
        "--seg_dir",
        help=(
            "Root of the directory where brain segmentations will be stored. "
            "If segmentations already exist, they will be used."
        ),
        required=True,
    )
    parser.add_argument(
        "--bids_csv",
        help="CSV file where the list of available LR series and masks will be stored.",
        default="bids_csv.csv",
    )
    parser.add_argument(
        "--iqms_csv",
        help="CSV file where the computed IQMs will be stored.",
        default="iqms_csv.csv",
    )

    parser.add_argument(
        "--out_csv",
        help="CSV file where the predictions from FetMRQC will be stored.",
        default="out_csv.csv",
    )

    parser.add_argument(
        "--fetmrqc20_iqms",
        help="Whether the IQMs from FetMRQC-20 should be computed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--use_all_iqms",
        help="Whether all IQMs should be computed",
        default=False,
        action="store_false",
        dest="fetmrqc20_iqms",
    )
    parser.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the MONAIfbs model.",
        default=BRAIN_CKPT,
    )

    parser.add_argument(
        "--mask_pattern",
        help=(
            "Pattern according to which the masks will be stored.\n "
            'By default, masks will be stored in "<masks_dir>/sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}'
            '[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz", and the different fields will be '
            "substituted based on the structure of bids_dir."
        ),
        type=str,
        default=MASK_PATTERN,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to control the randomization (to be used with randomize=True).",
    )

    parser.add_argument(
        "--device",
        help="Device to use for inference.",
        choices=["cpu", "cuda"],
        default="cuda",
    )


def build_reports_parser(parser):
    parser.add_argument(
        "--bids_dir",
        required=True,
        help="BIDS directory containing the LR series.",
    )

    parser.add_argument(
        "--masks_dir",
        help="Root of the BIDS directory where brain masks will be stored.",
        required=True,
    )

    parser.add_argument(
        "--reports_dir",
        help="Directory where the reports will be stored. (Default is <bids_dir>/derivatives/reports)",
        default=None,
    )

    parser.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the MONAIfbs model.",
        default=BRAIN_CKPT,
    )

    parser.add_argument(
        "--mask_pattern",
        help=(
            "Pattern according to which the masks will be stored.\n "
            'By default, masks will be stored in "<masks_dir>/sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}'
            '[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz", and the different fields will be '
            "substituted based on the structure of bids_dir."
        ),
        type=str,
        default=MASK_PATTERN,
    )
    parser.add_argument(
        "--bids_csv",
        help="CSV file where the list of available LR series and masks will be stored.",
        default="bids_csv.csv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
