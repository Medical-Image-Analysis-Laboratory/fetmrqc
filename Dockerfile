FROM nvcr.io/nvidia/pytorch:23.02-py3

ENV MAMBA_ROOT_PREFIX="/opt/micromamba" CONDA_PREFIX="/opt/micromamba"
RUN curl -L http://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xj -C /tmp bin/micromamba && \
    cp /tmp/bin/micromamba /bin/micromamba && \
    mkdir -p "$(dirname $MAMBA_ROOT_PREFIX)" && \
    /bin/micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX 

ENV PATH="$MAMBA_ROOT_PREFIX/bin:${PATH}"

# Create the nnUNet environment
RUN micromamba env create -n nnunet
RUN micromamba install -y -n nnunet python=3.9.15 -c conda-forge
RUN micromamba run -n nnunet python -m pip install git+https://github.com/MIC-DKFZ/nnUNet@af8c9fb1fe5c695020aa7c30c174a5e61efdd95d

# Setting up FetMRQC
RUN mkdir /fetmrqc
WORKDIR /fetmrqc

COPY fetal_brain_qc/ setup.py requirements.txt ./fetal_brain_qc/
RUN mv fetal_brain_qc/setup.py setup.py && \ 
    mv fetal_brain_qc/requirements.txt requirements.txt && \
    micromamba run -n base micromamba install -y python=3.9.15 -c conda-forge && \ 
    micromamba run -n base python -m pip install -e . && \
    mkdir -p "$MAMBA_ROOT_PREFIX/lib/python3.9/site-packages/monaifbs/config" && \
    wget -O "$MAMBA_ROOT_PREFIX/lib/python3.9/site-packages/monaifbs/config/monai_dynUnet_inference_config.yml" \
    "https://raw.githubusercontent.com/gift-surg/MONAIfbs/main/monaifbs/config/monai_dynUnet_inference_config.yml" && \
    chmod 644 fetal_brain_qc/models/MONAIfbs_dynunet_ckpt.pt
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]
