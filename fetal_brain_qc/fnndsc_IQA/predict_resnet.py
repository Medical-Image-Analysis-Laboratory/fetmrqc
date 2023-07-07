""" Image quality assessment based on FNNDSC's implementation.

Based on the code from Ivan Legorreta at
https://github.com/FNNDSC/pl-fetal-brain-assessment/blob/main/fetal_brain_assessment/predict_resnet.py

(c) 2021 Fetal-Neonatal Neuroimaging & Developmental Science Center
    Boston Children's Hospital
    http://childrenshospital.org/FNNDSC/
    dev@babyMRI.org
"""
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from .resnet_architecture import (
    model_architecture as create_model_architecture,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        weights="/usr/local/share/fetal_brain_assessment/weights_resnet.hdf5",
    ):

        logger.debug("Creating model")
        self.model = create_model_architecture()
        logger.debug("Model created")

        self.model.compile(
            loss=lambda y_true, y_pred: Huber(y_true, y_pred, delta=0.15),
            optimizer=Adam(learning_rate=0.0001),
            metrics=["mean_absolute_error"],
        )
        logger.debug("Model compiled")
        logger.debug("Loading resnet weights from %s", weights)
        self.model.load_weights(weights)
        logger.debug("Predictor object setup complete.")

    def predict(self, stacked_data, row_names) -> pd.DataFrame:
        stacked_data = np.array(stacked_data, dtype=np.float32)

        # Normalize dataset
        min1 = np.amin(stacked_data)
        max1 = np.amax(stacked_data)
        logger.info("Min: %s", min1)
        logger.info("Max: %s", max1)
        stacked_data = (stacked_data - min1) / (max1 - min1)
        min1 = np.amin(stacked_data)
        max1 = np.amax(stacked_data)
        logger.info("New min: %s", min1)
        logger.info("New max: %s", max1)
        logger.debug("Doing prediction")
        prediction = self.model.predict(
            stacked_data, verbose=1 if logger.level < 25 else 0
        )

        df = pd.DataFrame(row_names, columns=["filename"])
        df["quality"] = prediction
        return df
