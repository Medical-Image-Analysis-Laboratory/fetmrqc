from setuptools import setup, find_packages


def install_requires(fname="requirements.txt"):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


setup(
    name="fetal_brain_qc",
    version="0.1.3",
    packages=find_packages(),
    include_package_data=True,  
    package_data={
        "fetal_brain_qc": ["data/reports/*.html"],
    },
    description="Quality control for fetal brain MRI",
    author="Thomas Sanchez",
    author_email="thomas.sanchez@unil.ch",
    install_requires=install_requires(),
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
            "qc_reports_pipeline = fetal_brain_qc.cli.run_reports_pipeline:main",
            "qc_inference_pipeline = fetal_brain_qc.cli.run_inference_pipeline:main",
        ],
    },
)
