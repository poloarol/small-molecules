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

    def __init__(self, encoder: keras.Model, decoder: keras.Model) -> None:
        super().__init__()
        self.encoder: keras.Model = encoder
        self.decoder: keras.Model = decoder
        self.property_layer: layers.Dense = layers.Dense(1)
        self.train_total_loss: float = keras.metrics.Mean(name="train_total_loss")
        self.val_total_loss: float = keras.metrics.Mean(name="val_total_loss")

    @abstractmethod
    def train_step(self, datum) -> Dict[str, int]:
        """
        Training step simulation for VAE
        """

    @abstractmethod
    def test_step(self, datum) -> Dict[str, int]:
        """
        Testing step simulation for VAE
        """

    @abstractmethod
    def calculate_loss(self, **kwargs) -> float:
        """ calculate the loss at the specific training step """