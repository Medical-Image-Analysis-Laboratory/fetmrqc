# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
# This code was originally written by FNNDSC/BCH and is part of the fetal_brain_assessment
# repository at ttps://github.com/FNNDSC/pl-fetal-brain-assessment/blob/main/fetal_brain_assessment/predict_resnet.py
# (originally licensed under an MIT license)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Image quality assessment based on FNNDSC's implementation. """
import time
from tensorflow.keras.layers import (
    Conv3D,
    MaxPool3D,
    Dense,
    GlobalAveragePooling3D,
    Add,
)
from tensorflow.keras.layers import (
    Dropout,
    Input,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.models import Model

started_at = time.asctime()


# Identity Block for ResNet
def id_block(X, f):
    X_shortcut = X

    # First component of main path
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv3D(
        filters=int(f / 4),
        kernel_size=1,
        kernel_initializer="he_uniform",
        padding="same",
    )(X)

    # Second component of main path
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv3D(
        filters=int(f / 4),
        kernel_size=3,
        kernel_initializer="he_uniform",
        padding="same",
    )(X)

    # Third component of main path
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv3D(
        filters=f,
        kernel_size=1,
        kernel_initializer="he_uniform",
        padding="same",
    )(X)

    # Merge paths with skip connection
    X = Add()([X, X_shortcut])

    return X


def model_architecture():
    input_imgs = Input(shape=(217, 178, 60, 1))

    conv2 = Conv3D(
        filters=2,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(input_imgs)

    maxPool1 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)

    bn4 = BatchNormalization()(maxPool1)
    act4 = Activation("relu")(bn4)
    conv4 = Conv3D(
        filters=4,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act4)

    maxPool2 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv4)

    bn8 = BatchNormalization()(maxPool2)
    act8 = Activation("relu")(bn8)
    conv8 = Conv3D(
        filters=8,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act8)

    bn16 = BatchNormalization()(conv8)
    act16 = Activation("relu")(bn16)
    conv16 = Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act16)

    conv_block16 = id_block(conv16, 16)
    conv_block16 = id_block(conv_block16, 16)
    conv_block16 = id_block(conv_block16, 16)
    conv_block16 = id_block(conv_block16, 16)
    conv_block16 = id_block(conv_block16, 16)
    conv_block16 = id_block(conv_block16, 16)
    conv_block16 = id_block(conv_block16, 16)
    conv_block16 = id_block(conv_block16, 16)

    maxPool3 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_block16)

    bn32 = BatchNormalization()(maxPool3)
    act32 = Activation("relu")(bn32)
    conv32 = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act32)

    bn64 = BatchNormalization()(conv32)
    act64 = Activation("relu")(bn64)
    conv64 = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act64)

    maxPool4 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv64)

    bn128 = BatchNormalization()(maxPool4)
    act128 = Activation("relu")(bn128)
    conv128 = Conv3D(
        filters=128,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act128)

    bn256 = BatchNormalization()(conv128)
    act256 = Activation("relu")(bn256)
    conv256 = Conv3D(
        filters=256,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act256)

    bn512 = BatchNormalization()(conv256)
    act512 = Activation("relu")(bn512)
    conv512 = Conv3D(
        filters=512,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act512)

    conv_block512 = id_block(conv512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)
    conv_block512 = id_block(conv_block512, 512)

    bn1024 = BatchNormalization()(conv_block512)
    act1024 = Activation("relu")(bn1024)
    conv1024 = Conv3D(
        filters=1024,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act1024)

    conv_block1024 = id_block(conv1024, 1024)
    conv_block1024 = id_block(conv_block1024, 1024)
    conv_block1024 = id_block(conv_block1024, 1024)
    conv_block1024 = id_block(conv_block1024, 1024)
    conv_block1024 = id_block(conv_block1024, 1024)
    conv_block1024 = id_block(conv_block1024, 1024)

    bn2048 = BatchNormalization()(conv_block1024)
    act2048 = Activation("relu")(bn2048)
    conv2048 = Conv3D(
        filters=2048,
        kernel_size=(3, 3, 3),
        kernel_initializer="he_uniform",
        padding="same",
    )(act2048)

    conv_block2048 = id_block(conv2048, 2048)
    conv_block2048 = id_block(conv_block2048, 2048)
    conv_block2048 = id_block(conv_block2048, 2048)
    conv_block2048 = id_block(conv_block2048, 2048)
    conv_block2048 = id_block(conv_block2048, 2048)
    conv_block2048 = id_block(conv_block2048, 2048)

    conv_block2048 = BatchNormalization()(conv_block2048)
    conv_block2048 = Activation("relu")(conv_block2048)

    globalAP = GlobalAveragePooling3D()(conv_block2048)
    dense1 = Dense(
        units=256, activation="relu", kernel_initializer="he_uniform"
    )(globalAP)
    dp = Dropout(0.4)(dense1)
    dense2 = Dense(units=1, activation="linear")(dp)

    model = Model(inputs=input_imgs, outputs=dense2)
    return model
