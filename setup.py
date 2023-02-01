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
            "qc_generate_index = fetal_brain_qc.cli.generate_index:main",
            "qc_randomize_reports = fetal_brain_qc.cli.randomize_reports:main",
            "qc_generate_reports = fetal_brain_qc.cli.generate_reports:main",
            "qc_list_bids_csv = fetal_brain_qc.cli.run_list_and_anon_bids:main",
            "qc_run_pipeline = fetal_brain_qc.cli.run_pipeline:main",
            "qc_brain_extraction = fetal_brain_qc.cli.run_brain_extraction:main",
            "qc_preprocessing = fetal_brain_qc.cli.run_preprocessing:main",
            "qc_fetal_iqa_mit = fetal_brain_qc.cli.run_iqa:main",
            "qc_fetal_iqa_fnndsc = fetal_brain_qc.cli.run_stack_iqa:main",
            "qc_compute_metrics = fetal_brain_qc.cli.run_qc:main",
            "qc_ratings_to_csv = fetal_brain_qc.cli.ratings_to_csv:main",
        ],
    },
)
