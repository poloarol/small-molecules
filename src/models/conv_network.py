""" conv_network.py """

from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RelationalGraphLayer(layers.Layer):
    """ """

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
            shape = (bond_dim, atom_dim, self.units),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            trainable = True,
            name = "Weights",
            dtype = tf.float32
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape = (bond_dim, 1, self.units),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                trainable = True,
                name = "bias",
                dtype = tf.float32
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
