""" gvae.py """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .converter import DescriptorsVAE, GraphConverterVAE
from .network_utils import Sampling
from .vae import VAE


class GraphVAE(keras.Model):
    """
    Graph VAE
    """

    def __init__(
        self, encoder: keras.Model, decoder: keras.Model, latent_dim: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.encoder: keras.Model = encoder
        self.decoder: keras.Model = decoder
        self.latent_dim = latent_dim
        self.property_prediction_layer = layers.Dense(1)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def _compute_loss(
        self, z_mean, z_log_var, qed_true, qed_predicted, real_graph, predicted_graph
    ) -> float:

        adjacency_real, features_real = real_graph
        adjacency_generated, features_generated = predicted_graph

        adjacency_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(
                    adjacency_real, adjacency_generated
                ),
                axis=(1, 2),
            )
        )
        feature_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(
                    features_real, features_generated
                ),
                axis=(1),
            )
        )

        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.math.exp(z_log_var), 1
        )
        kl_loss = tf.reduce_mean(kl_loss)

        property_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(qed_true, qed_predicted)
        )

        graph_loss = self._gradient_penalty(real_graph, predicted_graph)
        self.kl_loss_tracker.update_state(kl_loss)

        return kl_loss + property_loss + graph_loss + adjacency_loss + feature_loss

    def _gradient_penalty(self, real_graph, predicted_graph):
        # Unpack graphs

        adjacency_real, features_real = real_graph
        adjacency_generated, features_generated = predicted_graph

        # Generated interpolated graphs
        alpha = tf.random.uniform([self.batch_size])
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        adjacency_interp = (adjacency_real * alpha) + (1 - alpha) * adjacency_generated
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
        features_interp = (features_real * alpha) + (1 - alpha) * features_generated

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(adjacency_interp)
            tape.watch(features_interp)

            _, _, logits, _, _ = self(
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
        # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # return reconstruction_loss

    def call(self, inputs):
        z_mean, log_var = self.encoder(inputs)
        latent_space = Sampling()([z_mean, log_var])

        adjacency_generated, generated_features = self.decoder(latent_space)
        property_predicted = self.property_prediction_layer(z_mean)

        return (
            z_mean,
            log_var,
            property_predicted,
            adjacency_generated,
            generated_features,
        )

    def inference(self, batch_size: int):
        latent_space = tf.random.normal((batch_size, self.latent_dim))
        reconstruction_adjacency, reconstruction_features = self.decoder(latent_space)
        # Obtain one-hot encoded adjacency tensor
        adjacency = tf.argmax(reconstruction_adjacency, axis=1)
        adjacency = tf.one_hot(adjacency, depth=DescriptorsVAE.BOND_DIM.value, axis=1)
        # Remove potential self-loops from adjacency
        adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
        # Obtain one-hot encoded tensor
        features = tf.argmax(reconstruction_features, axis=2)
        features = tf.one_hot(features, depth=DescriptorsVAE.ATOM_DIM.value, axis=2)

        return [
            GraphConverterVAE(adjacency[i].numpy(), features[i].numpy()).transform()
            for i in range(batch_size)
        ]

    def train_step(self, datum):
        adjacency_tensor, feature_tensor, qed_tensor = datum[0]
        graph_real = [adjacency_tensor, feature_tensor]
        self.batch_size = tf.shape(qed_tensor)[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, qed_pred, generator_adjacency, generator_feature = self(
                graph_real, training=True
            )
            graph_generated = [generator_adjacency, generator_feature]
            total_loss = self._compute_loss(
                z_mean=z_mean,
                z_log_var=z_log_var,
                qed_true=qed_tensor,
                qed_predicted=qed_pred,
                real_graph=graph_real,
                predicted_graph=graph_generated,
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total-loss": self.total_loss_tracker.result(),
            "kl-loss": self.kl_loss_tracker.result(),
            # "reconstruction-loss": self.reconstruction_loss_tracker.result(),
        }
