""" gan.py """

from abc import abstractmethod
from typing import Dict, Final

import tensorflow as tf
from tensorflow import keras

class GAN(keras.Model):
    """
    Abstract class to build a
    Generative Adversarial Network.
    """

    def __init__(
        self,
        discriminator_model: keras.Model,
        generator_model: keras.Model,
        discriminator_steps: int = 1,
        generator_steps: int = 1,
        gp_weight: float = 10.0,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.discriminator: keras.Model = discriminator_model
        self.generator: keras.Model = generator_model
        self.discriminator_steps: int = discriminator_steps
        self.generator_steps: int = generator_steps
        self.gp_weight: float = gp_weight
        self.latent_dim: Final[int] = latent_dim

    def compile(self, generator_opt: float, discriminator_opt: float, **kwargs):
        """Compile the model"""
        super().compile(**kwargs)
        self.optimizer_generator: float = generator_opt
        self.optimizer_discriminator: float = discriminator_opt
        self.metric_generator = keras.metrics.Mean(name="loss_gen")
        self.metric_discriminator = keras.metrics.Mean(name="loss_dis")

    @abstractmethod
    def train_step(self, inputs, **kwargs) -> Dict:
        """provides methods to train the model"""

    @abstractmethod
    def _loss_discriminator(self, real_input, fake_input) -> float:
        """Calculate the loss of the discriminator"""

    @abstractmethod
    def _loss_generator(self, generated_input) -> float:
        """Calculate the loss of the discriminator"""

if __name__ == "__main__":
    pass
