""" wgan.py """

from typing import Dict, Final

import tensorflow as tf
from .converter import Descriptors
from .gan import GAN
from .network_utils import build_graph_discriminator, build_graph_generator
from tensorflow import keras


class GraphWGAN(GAN):
    """
    Wasserstein Generative Adversarial Network with Gaussian Process,
    (gWGAN-GP) with Relational Convolutional Graph Neural Network (RGCN)
    """

    def __init__(
        self,
        discriminator_model: keras.Model,
        generator_model: keras.Model,
        batch_size: int = 32,
    ) -> None:
        super().__init__(discriminator_model, generator_model)
        self.batch_size: int = batch_size

    def compile(self, generator_opt: float, discriminator_opt: float, **kwargs):
        """Compile the model"""
        super().compile(**kwargs)
        self.optimizer_generator: float = generator_opt
        self.optimizer_discriminator: float = discriminator_opt
        self.metric_generator = keras.metrics.Mean(name="loss_gen")
        self.metric_discriminator = keras.metrics.Mean(name="loss_dis")


    def _loss_discriminator(self, real_input, generated_input) -> float:
        logits_real = self.discriminator(real_input, training=True)
        logits_generated = self.discriminator(generated_input, training=True)
        loss: float = tf.reduce_mean(logits_generated) - tf.reduce_mean(logits_real)
        loss_gp: float = self._gradient_penalty(real_input, generated_input)

        return loss + loss_gp * self.gp_weight

    def _gradient_penalty(self, real_input, generated_input):
        # Unpack graph
        adjacency_real, features_real = real_input
        adjacency_generated, features_generated = generated_input

        # Generate interpolated grapsh (adjacency_interp and features_interp)
        alpha = tf.random.uniform([self.batch_size])
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        adjacency_interp = (adjacency_real * alpha) + (1 - alpha) * adjacency_generated
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
        features_interp = (features_real * alpha) + (1 - alpha) * features_generated

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(adjacency_interp)
            tape.watch(features_interp)
            logits = self.discriminator(
                [adjacency_interp, features_interp], training=True
            )

        # Compute the gradients w.r.t the interpolated graphs
        grads = tape.gradient(logits, [adjacency_interp, features_interp])
        # Compute the gradient penalty
        grads_adjacency_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
        grads_features_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
        return tf.reduce_mean(
            tf.reduce_mean(grads_adjacency_penalty, axis=(-2, -1))
            + tf.reduce_mean(grads_features_penalty, axis=(-1))
        )

    def _loss_generator(self, generated_input) -> float:
        logits_generated = self.discriminator(generated_input, training=True)
        return -tf.reduce_mean(logits_generated)

    def train_step(self, inputs) -> Dict:
        """
        GAN training step
        """
        if isinstance(inputs[0], tuple):
            inputs = inputs[0]

        graph_real = inputs

        # Train the discriminator for one or more steps
        for _ in range(self.discriminator_steps):
            latent_space = tf.random.normal((self.batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                graph_generated = self.generator(latent_space, training=True)
                loss = self._loss_discriminator(graph_real, graph_generated)

            grads = tape.gradient(loss, self.discriminator.trainable_weights)
            self.optimizer_discriminator.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
            self.metric_discriminator.update_state(loss)

        # Train the generator for one or more steps
        for _ in range(self.generator_steps):
            latent_space = tf.random.normal((self.batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                graph_generated = self.generator(latent_space, training=True)
                loss = self._loss_generator(graph_generated)

                grads = tape.gradient(loss, self.generator.trainable_weights)
                self.optimizer_generator.apply_gradients(
                    zip(grads, self.generator.trainable_weights)
                )
                self.metric_generator.update_state(loss)

        return {metric.name: metric.result() for metric in self.metrics}


if __name__ == "__main__":

    LATENT_DIM: Final[int] = 64
    generator = build_graph_generator(
        dense_units=[128, 256, 512],
        droupout_rate=0.2,
        latent_dim=LATENT_DIM,
        adjacency_shape=(
            Descriptors.BOND_DIM.value,
            Descriptors.NUM_ATOMS.value,
            Descriptors.NUM_ATOMS.value,
        ),
        feature_shape=(Descriptors.NUM_ATOMS.value, Descriptors.NUM_ATOMS.value),
    )
    discriminator = build_graph_discriminator(
        gconv_units=[128, 128, 128, 128],
        dense_units=[512, 512],
        droupout_rate=0.2,
        adjacency_shape=(
            Descriptors.BOND_DIM.value,
            Descriptors.NUM_ATOMS.value,
            Descriptors.NUM_ATOMS.value,
        ),
        feature_shape=(Descriptors.NUM_ATOMS.value, Descriptors.BOND_DIM.value),
    )

    generator.summary()
    discriminator.summary()

    wgan = GraphWGAN(generator, discriminator)

    wgan.compile(
        generator_opt=keras.optimizers.Adam(5e-4),
        discriminator_opt=keras.optimizers.Adam(5e-4),
    )
