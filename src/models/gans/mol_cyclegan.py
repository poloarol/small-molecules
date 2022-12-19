""" mol_cyclegan.py """

from typing import Final

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gan import GAN
from network_utils import cycle_gan_discriminator, cycle_gan_generator


class CycleGAN(GAN):
    """ Molecular CycleGAN, for molecular generation.

    Args:
        GAN (Object): Abstract class to build GAN models.
    """
    
    def __init__(self, discriminator_model: keras.Model, generator_model: keras.Model, batch_size: int = 32):
        super().__init__(discriminator_model, generator_model)
        self.batch_size: int = batch_size

    def _gan_criterion(self, output, target, use_lsgan: bool = True, use_wasserstein: bool = False) -> float:
        assert np.sum(
            ([use_lsgan, use_wasserstein]) < 2,
            "You can only use one of these parameters"
        )
        loss: float = 0.0
        epsilon: Final[float] = 1e-6
        if use_wasserstein:
            loss = output * target
        elif use_lsgan:
            loss = (output - target) ** 2
        else:
            loss = (tf.math.log(output + epsilon) * target) + tf.math.log(1 - output + epsilon) * (1 - target)

        dimensions: int = list(range(1, tf.experimental.numpy.ndim(loss)))
        return tf.expand_dims((tf.experimental.numpy.mean(loss, dimensions)), 0)

    def _cycle_criterion(real, fake, use_abs: bool = True) -> float:
        difference: float = 0.0
        if use_abs:
            difference = tf.math.abs(fake - real)
        else:
            difference = (fake - real) ** 2
        dimensions: int = list(range(1, tf.experimental.numpy.ndim(difference)))
        return tf.expand_dims((tf.experimental.numpy.mean(dimensions, dimensions)), 0)

    def _compute_similarity_loss(self, real_input, real_ouput, generated_input, generated_output):
        loss_x: float = self._cycle_criterion(real_input, real_ouput)
        loss_y: float = self._cycle_criterion(generated_input, generated_output)
        
        return loss_x + loss_y

    def _loss_generator(self, tensor, cycle_loss_weight: float = 0.3, loss_weight: float = 0.1, use_wgan: bool = False):
        network_A_predict_fake, rec_A, real_A, network_B_predict_fake, rec_B, real_B, fake_A, fake_B = tensor
        
        # GAN loss
        first_target = None
        second_target = None

        if use_wgan:
            first_target = -tf.ones_like(network_A_predict_fake, dtype=tf.float32)
            second_target = -tf.ones_like(network_B_predict_fake, dtype=tf.float32)
        else:
            first_target = tf.ones_like(network_A_predict_fake, dtype=tf.float32)
            second_target = tf.ones_like(network_B_predict_fake, dtype=tf.float32)

        first_loss: float = self._gan_criterion(network_A_predict_fake, first_target)
        second_loss: float = self._gan_criterion(network_B_predict_fake, second_target)
        
        gan_loss: float = (first_loss + second_loss) * cycle_loss_weight

        # Cycle loss
        cycle_loss: float = self._compute_similarity_loss(rec_A, real_A, rec_B, real_B)
        # Identity loss
        identity_loss: float = self._compute_similarity_loss(real_A, real_B, fake_A, fake_B)
        
        loss: float = gan_loss + cycle_loss + (identity_loss * loss_weight)
        
        return loss

    def discriminator_loss(self, prediction, use_wgan: bool = False) -> float:
        real, fake = prediction
        real_target = None
        fake_target = None

        if use_wgan:
            real_target = -tf.ones_like(real, dtype=tf.float32)
            fake_target = -tf.ones_like(fake, dtype=tf.float32)
        else:
            real_target = tf.ones_like(real, dtype=tf.float32)
            fake_target = tf.ones_like(fake, dtype=tf.float32)

        real_loss: float = self._gan_criterion(real, real_loss)
        fake_loss: float = self._gan_criterion(fake, fake_loss)
        
        loss: float = 0.5 * (real_loss + fake_loss)
        return loss      
            