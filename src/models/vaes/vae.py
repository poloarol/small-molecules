""" vae.py """

from abc import abstractmethod
from typing import Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class VAE(keras.Model):
    """
    Abstract class to build a
    Variational Adversarial Network
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model, max_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder: keras.Model = encoder
        self.decoder: keras.Model = decoder
        self.max_length: int = max_length
        self.property_prediction_layer = layers.Dense(1)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

    @abstractmethod
    def train_step(self, datum) -> Dict[str, int]:
        """
        Training step simulation for VAE
        """

    @abstractmethod
    def _compute_loss(self, **kwargs):
        """
        Compute the loss of the VAE
        """

    @abstractmethod
    def _gradient_penalty(self, **kwargs):
        """
        Compute the gradient penalty
        """

    @abstractmethod
    def inference(self, **kwargs):
        """
        Sample the latent space
        """

    @abstractmethod
    def call(self, inputs):
        """
        Get properties
        """