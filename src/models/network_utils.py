""" conv_network.py """

from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RelationalConvGraphLayer(layers.Layer):
    """
    The Relational Convolutional Graph Layers implements
    non-linearly transformed neighbourhood aggregations
    """

    def __init__(
        self,
        units: int = 128,
        activation: str = "relu",
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Any = None,
        bias_regularizer: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units: int = units
        self.activation: str = keras.activations.get(activation)
        self.use_bias: bool = use_bias
        self.bias_initializer: Any = keras.initializers.get(bias_initializer)
        self.kernel_regularizer: Any = keras.regularizers.get(kernel_regularizer)
        self.kernel_initializer: Any = keras.initializers.get(kernel_initializer)
        self.bias_regularizer: Any = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape) -> None:
        bond_dim: int = input_shape[0][1]
        atom_dim: int = input_shape[1][2]

        self.kernel = self.add_weight(
            shape=(bond_dim, atom_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="Weights",
            dtype=tf.float32,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(bond_dim, 1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="bias",
                dtype=tf.float32,
            )

    def call(self, inputs):
        adjacency, features = inputs
        # Aggreage informaton from neighbours
        data_point = tf.matmul(adjacency, features[:, None, :, :])
        # Apply linear transformation
        data_point = tf.matmul(data_point, self.kernel)

        if self.use_bias:
            data_point = data_point + self.bias
            # reduce bond types dim
            dp_reduced = tf.reduce_sum(data_point, axis=1)
            # apply non-linear transformaton
            return self.activation(dp_reduced)

        return data_point

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


if __name__ == "__main__":
    pass