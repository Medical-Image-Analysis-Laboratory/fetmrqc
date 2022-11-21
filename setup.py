from setuptools import setup, find_packages

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
            "qc_generate_reports = fetal_brain_qc.cli.generate_reports:main",
            "qc_list_bids_csv = fetal_brain_qc.cli.run_list_and_anon_bids:main",
            "qc_run_pipeline = fetal_brain_qc.cli.run_pipeline:main",
        ],
    },
)
