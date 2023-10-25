from setuptools import setup

setup(
    name="fetal_brain_qc",
    version="0.0.1",
    packages=["fetal_brain_qc"],
    description="Quality control for fetal brain MRI",
    author="Thomas Sanchez",
    author_email="thomas.sanchez@unil.ch",
    entry_points={
        "console_scripts": [
            "qc_list_bids_csv = fetal_brain_qc.cli.list_and_anon_bids:main",
            "qc_generate_index = fetal_brain_qc.cli.generate_index:main",
            "qc_generate_reports = fetal_brain_qc.cli.generate_reports:main",
            "qc_brain_extraction = fetal_brain_qc.cli.brain_extraction:main",
            "qc_segmentation = fetal_brain_qc.cli.compute_segmentation:main",
            "qc_compute_iqms = fetal_brain_qc.cli.compute_iqms:main",
            "qc_niftymic_qc = fetal_brain_qc.cli.qc_niftymic:main",
            "qc_ratings_to_csv = fetal_brain_qc.cli.ratings_to_csv:main",
            "qc_inference = fetal_brain_qc.cli.inference:main",
            "qc_training = fetal_brain_qc.cli.training:main",
        ],
    },
)
