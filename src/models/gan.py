""" gan.py """

from abc import ABC, abstractmethod
from typing import Dict, Final

import tensorflow as tf
from network_utils import RelationalConvGraphLayer
from tensorflow import keras
from tensorflow.keras import layers

from utils.converter import Descriptors


def build_graph_generator(
    dense_units: int,
    droupout_rate: float,
    latent_dim: int,
    adjacency_shape: int,
    feature_shape: int,
) -> keras.Model:
    """
    Build a GAN generator model
    """

    z_space = layers.Input(shape=(latent_dim,))
    # Propagate through one or more densely connected layers
    latent_space = z_space
    for unit in dense_units:
        latent_space = layers.Dense(unit, activation="tanh")(latent_space)
        latent_space = layers.Dropout(droupout_rate)(latent_space)

    # Map outputs of previous layer (x_space) to
    # [continous] adjacency tensors (x_adjacency)
    x_adjacency = layers.Dense(tf.math.reduce_prod(adjacency_shape))(latent_space)
    x_adjacency = layers.Reshape(adjacency_shape)(x_adjacency)
    # Symmetrify tensors in the last two dimensions
    x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 2, 3))) / 2
    x_adjacency = layers.Softmax(axis=1)(x_adjacency)

    # Map outputs of previous layer (x)
    # to [continuous] feature tensors (x_features)
    x_features = layers.Dense(tf.math.reduce_prod(feature_shape))(latent_space)
    x_features = layers.Reshape(feature_shape)(x_features)
    x_features = layers.Softmax(axis=2)(x_features)

    model = keras.Model(
        inputs=z_space, outputs=[x_adjacency, x_features], name="Generator"
    )
    return model


def build_graph_discriminator(
    gconv_units: int,
    dense_units: int,
    droupout_rate: float,
    adjacency_shape: int,
    feature_shape: int,
) -> keras.Model:
    """
    Build a graph discriminator for a GAN
    """

    adjacency = layers.Input(shape=adjacency_shape)
    features = layers.Input(shape=feature_shape)

    # Propagate through one or more graph convolutional layers
    features_transformed = features
    for unit in gconv_units:
        features_transformed = RelationalConvGraphLayer(units=unit)(
            [adjacency, features_transformed]
        )

    # Reduce 2-D representation of molecule to 1-D
    x_space = layers.GlobalAveragePooling1D()(features_transformed)

    # Propagate through one or more densely connected layers
    for unit in dense_units:
        x_space = layers.Dense(unit, activation="relu")(x_space)
        x_space = layers.Dropout(droupout_rate)(x_space)

    # For each molecule, output a single scalar value expressing
    # the "realness" of the inputted molecule

    x_out = layers.Dense(inputs=[adjacency, features])(x_space)
    model = keras.Model(inputs=[adjacency, features], outputs=x_out)

    return model


class GAN(ABC):
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
    def train_step(self, inputs) -> Dict:
        """provides methods to train the model"""

    @abstractmethod
    def _loss_discriminator(self, real_input, fake_input) -> float:
        """Calculate the loss of the discriminator"""

    def _loss_generator(self, generated_input) -> float:
        """Calculate the loss of the discriminator"""
        logits_generated = self.discriminator(generated_input, training=True)
        return -tf.reduce_mean(logits_generated)


class GraphWGAN(GAN):
    """
    Wasserstein Generative Adversarial Network,
    with Relational Convolutional Graph Neural Network
    """

    def __init__(
        self,
        discriminator_model: keras.Model,
        generator_model: keras.Model,
        batch_size: int = 32,
    ) -> None:
        super().__init__(discriminator_model, generator_model)
        self.batch_size: int = batch_size

    def _loss_discriminator(self, real_input, fake_input) -> float:
        logits_real = self.discriminator(real_input, training=True)
        logits_fake = self.discriminator(fake_input, training=True)
        loss: float = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
        loss_gp: float = self._gradient_penalty(real_input, fake_input)

        return loss + loss_gp * self.gp_weight

    def _gradient_penalty(self, real_input, fake_input):
        # Unpack graph
        adjacency_real, features_real = real_input
        adjacency_fake, features_fake = fake_input

        # Generate interpolated grapsh (adjacency_interp and features_interp)
        alpha = tf.random.uniform(self.batch_size)
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        adjacency_interp = (adjacency_real * alpha) + (1 - alpha) * adjacency_fake
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
        features_interp = (features_real * alpha) + (1 - alpha) * features_fake

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

    def train_step(self, inputs) -> Dict:

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
                graph_generated = self.generator(latent_space, self.latent_dim)
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
            Descriptors.BOND_DIM,
            Descriptors.NUM_ATOMS,
            Descriptors.NUM_ATOMS,
        ),
        feature_shape=(Descriptors.NUM_ATOMS, Descriptors.NUM_ATOMS),
    )
    discriminator = build_graph_discriminator(
        gconv_units=[128, 128, 128, 128],
        dense_units=[512, 512],
        droupout_rate=0.2,
        adjacency_shape=(
            Descriptors.BOND_DIM,
            Descriptors.NUM_ATOMS,
            Descriptors.NUM_ATOMS,
        ),
        feature_shape=(Descriptors.NUM_ATOMS, Descriptors.NUM_ATOMS),
    )

    generator.summary()
    discriminator.summary()

    wgan = GraphWGAN(generator, discriminator)

    wgan.compile(
        generator_opt=keras.optimizers.Adam(5e-4),
        discriminator_opt=keras.optimizers.Adam(5e-4),
    )
