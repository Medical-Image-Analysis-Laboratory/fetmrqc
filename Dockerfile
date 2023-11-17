FROM nvcr.io/nvidia/pytorch:23.02-py3
MAINTAINER Thomas Sanchez

ENV CONDA_PREFIX=/root/micromamba
RUN curl -L http://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xj -C /tmp bin/micromamba && \
    cp /tmp/bin/micromamba /bin/micromamba 

# Setup and update micromamba 
RUN mkdir -p "$(dirname $CONDA_PREFIX)" && \
    /bin/micromamba shell init -s bash -p ~/micromamba 
ENV PATH="$CONDA_PREFIX/bin:${PATH}"

# Create the nnUNet environment
RUN micromamba env create -n nnunet
RUN micromamba install -n nnunet python=3.9.15 -c conda-forge
RUN micromamba run -n nnunet python -m pip install git+https://github.com/MIC-DKFZ/nnUNet@af8c9fb1fe5c695020aa7c30c174a5e61efdd95d

# Setting up FetMRQC
RUN mkdir /fetmrqc
WORKDIR /fetmrqc
COPY . .
RUN micromamba run -n base micromamba install -y python=3.9.15 -c conda-forge
RUN micromamba run -n base python -m pip install -e .
# Download a missing config file
RUN mkdir "$CONDA_PREFIX/lib/python3.9/site-packages/monaifbs/config" && wget -O "$CONDA_PREFIX/lib/python3.9/site-packages/monaifbs/config/monai_dynUnet_inference_config.yml" https://raw.githubusercontent.com/gift-surg/MONAIfbs/main/monaifbs/config/monai_dynUnet_inference_config.yml

# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]
