""" mol_cyclegan.py """

import itertools
from typing import Final

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from gan import GAN
from network_utils import cycle_gan_discriminator, cycle_gan_generator


class CycleGAN(GAN):
    """ Molecular CycleGAN, for molecular generation.

    Args:
        GAN (Object): Abstract class to build GAN models.
    """
    
    def __init__(self, discriminator_model: keras.Model, generator_model: keras.Model, batch_size: int = 32, epochs: int = 100, **kwargs):
        super().__init__(discriminator_model, generator_model)
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.iters: int = 5
        self.patience: int = 1
        self.data_pooling: bool = True
        self.use_wgan: bool = False

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

    def _loss_discrimination(self, prediction, use_wgan: bool = False) -> float:
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

    def _train_functions(self, inputs, loss_function, layer_inputs):
        adam = Adam(lr=1e-4, beta_1=0.5, beta_2=0.999, epsilon=1e-7, decay=0.0)
        t_functions = keras.Model(inputs, layers.Lambda(loss_function)(layer_inputs))
        t_functions.compile(adam, "mse")
        return t_functions

    def __generator_train_function(self,
                                  discriminator_tuple, generator_tuple,
                                  real_datapoints, fake_datapoints,
                                  loss_weights, use_wgan: bool = False):
        first_discriminator_network, second_discriminator_network = discriminator_tuple
        first_generator_network, second_generator_network = generator_tuple
        real_input, real_output = real_datapoints
        fake_input, fake_output = fake_datapoints
        cycle_loss_weight, id_loss_weight = loss_weights
        
        first_fake_predictions = second_discriminator_network(fake_output)
        second_generated_dp = second_generator_network(fake_output)
        
        second_fake_predictions = first_discriminator_network(fake_input)
        first_generated_dp = first_generator_network(fake_input)
        
        lamba_layers = [first_fake_predictions, second_generated_dp, real_input,\
            second_fake_predictions, first_generated_dp, real_output,\
                fake_input, fake_output]
        
        for layer in itertools.chain(first_generator_network.layers, second_generator_network.layers):
            layer.trainable = True

        for layer in itertools.chain(first_discriminator_network.layers, second_discriminator_network.layers):
            layer.trainable = False
            if isinstance(layer, layers.BatchNormalization):
                layer._per_input_updates = {}

        net_generative_partial_loss: float = lambda x : self._loss_generator(x,
                                                                             cycle_loss_weight,
                                                                             id_loss_weight,
                                                                             use_wgan=use_wgan)

        generator_train_function = self._train_functions(inputs=[real_input, real_output],
                                                         loss_function=net_generative_partial_loss,
                                                         layer_inputs=lamba_layers)
        return generator_train_function
    
    def __first_discriminator_train_function(self, 
                                             discriminator_tuple, generator_tuple,
                                             real_datapoints, input_shape,
                                             use_wgan: bool = False):
        first_discriminator_model, second_discriminator_model = discriminator_tuple
        first_generator_model, second_generator_model = generator_tuple
        real_input, real_output = real_datapoints
        
        first_discriminator_prediction = first_discriminator_model(real_input)
        layer_shape = layers.Input(shape=input_shape)
        second_discriminator_prediction = first_discriminator_model(layer_shape)
        
        for layer in first_discriminator_model.layers:
            layer.trainable = True
        
        for layer in itertools.chain(first_generator_model.layers, second_generator_model.layers, second_discriminator_model.layers):
            layer.trainable = False
            if isinstance(layer, layers.BatchNormalization):
                layer._per_input_updates = {}

        net_discriminator_loss: float = lambda x : self._loss_discrimination(x, use_wgan=use_wgan)
        network_discriminator_train_function = self._train_functions(
            inputs=[real_input, real_output],
            loss_function=net_discriminator_loss,
            layer_inputs=[first_discriminator_prediction, second_discriminator_prediction]
        )
        
        return network_discriminator_train_function

    def __second_discriminator_train_function(self,
                                              discriminator_tuple, generator_tuple,
                                             real_datapoints, input_shape,
                                             use_wgan: bool = False):

        first_discriminator_model, second_discriminator_model = discriminator_tuple
        first_generator_model, second_generator_model = generator_tuple
        real_input, real_output = real_datapoints
        
        first_discriminator_prediction = second_discriminator_model(real_output)
        layer_shape = layers.Input(shape=input_shape)
        second_discriminator_prediction = second_discriminator_model(layer_shape)
        
        for layer in second_discriminator_model.layers:
            layer.trainable = True
        
        for layer in itertools.chain(first_generator_model.layers, second_generator_model.layers, first_discriminator_model.layers):
            layer.trainable = False
            if isinstance(layer, layers.BatchNormalization):
                layer._per_input_updates = {}

        net_discriminator_loss: float = lambda x : self._loss_discrimination(x, use_wgan=use_wgan)
        network_discriminator_train_function = self._train_functions(
            inputs=[real_input, real_output],
            loss_function=net_discriminator_loss,
            layer_inputs=[first_discriminator_prediction, second_discriminator_prediction]
        )
        
        return network_discriminator_train_function

    def _clip_weights(self, network, clip_lambda: float = 0.1):
        weights = [np.clip(weight, -clip_lambda, clip_lambda) for weight in network.get_weights()]
        network.set_weights(weights)

    def train_step(self,
                   inputs, batches_tuple,
                   generator_tuple, discriminator_tuple):
        generator_train_function = self.__generator_train_function()
        first_discriminator_train_function = self.__first_discriminator_train_function()
        second_discriminator_train_function = self.__second_discriminator_train_function()
        first_generator_function, second_generator_function = generator_tuple
        first_discriminator, second_discriminator = discriminator_tuple
        pool_one, pool_two = inputs
        train_batch, test_batch = batches_tuple

        for _, epoch in enumerate(self.epochs):
            target_label = np.zeros((self.batch_size, 1))
            epoch_count, datapoint_one, datapoint_two = next(train_batch)
            
            tmp_fake_two = first_generator_function([datapoint_one, 1])[0]
            tmp_fake_two = second_generator_function([datapoint_two, 1])[0]
            
            if self.use_wgan:
                pass
            else:
                pass