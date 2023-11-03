# Contains pytorch, torchvision, cuda, cudnn
FROM nvcr.io/nvidia/pytorch:22.04-py3
MAINTAINER Thomas Sanchez

#nnUnet is set up by default to look at the following dir. I mantain them for reproducibility
#ARG resources="/opt/nnunet_resources"
#ENV nnUNet_raw=$resources"/nnUNet_raw" nnUNet_preprocessed=$resources"/nnUNet_preprocessed" nnUNet_results=$resources"/nnUNet_results"

#ENV DEBIAN_FRONTEND=noninteractive


# Update Python to version 3.9 using Conda
RUN conda install -y python=3.9.15
RUN conda create --name nnunet --clone base


#Copy the files. TODO change to git clone https://github.com/PeterMcGor/nnUNet.git  once the repo is tested the repo is tested
RUN mkdir /fetmrqc
WORKDIR /fetmrqc
COPY . .
 
## Install nnunet 
RUN pip install -e .
