""" conv_network.py """

from typing import Any, Dict

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units: int = units
        self.use_bias: bool = use_bias
        self.activation: Any = keras.activations.get(activation)
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

    x_out = layers.Dense(1, dtype="float32")(x_space)
    model = keras.Model(
        inputs=[adjacency, features], outputs=x_out, name="Discriminator"
    )

    return model


def __activation_layer(use_leaky_relu: bool = True, leaky_alpha: float = 0.1):
    if use_leaky_relu:
        return layers.LeakyReLU(alpha=leaky_alpha)
    return layers.Activation("relu")


def __dense_layer(
    inputs, units: int = 56, use_batch_norm: bool = True, use_leaky_relu: bool = True
):
    x_input = layers.Dense(units=units, activation=None)(inputs)
    if use_batch_norm:
        x_input = layers.BatchNormalization()(x_input)
    return __activation_layer(use_leaky_relu=use_leaky_relu)(x_input)


def __residual_dense_block(
    inputs,
    units: int = 56,
    use_dropout: bool = False,
    use_batch_norm: bool = True,
    use_leaky_relu: bool = False,
):
    layer_output = __dense_layer(
        inputs,
        units=units,
        use_batch_norm=use_batch_norm,
        use_leaky_relu=use_leaky_relu,
    )

    if use_dropout:
        layer_output = layers.Dropout(0.5)(layer_output)

    layer_output = layers.Dense(units=units, activation=None)(layer_output)

    if use_batch_norm:
        layer_output = layers.BatchNormalization()(layer_output)

    layer_output = layers.Add()([layer_output, inputs])
    layer_output = __activation_layer(use_leaky_relu=use_leaky_relu)(layer_output)

    return layer_output


def cycle_gan_discriminator(
    input_shape=(56,), nn_size: str = "big"
) -> Dict[str, keras.Model]:
    """_summary_

    Args:
        input_shape (tuple, optional): _description_. Defaults to (56,).
        nn_size (str, optional): _description_. Defaults to "bigger".
    """

    inputs = layers.Input(input_shape)

    def big_layer(
        use_wgan: bool = False,
        use_batch_norm: bool = False,
        use_leaky_relu: bool = False,
    ):
        x_input = __dense_layer(
            inputs,
            units=42,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=42,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )

        x_input = __dense_layer(
            inputs,
            units=28,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=28,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )

        x_input = __dense_layer(
            inputs,
            units=14,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=14,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )

        x_input = __dense_layer(
            inputs,
            units=7,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=7,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )

        activation = None if use_wgan else "sigmoid"
        output = layers.Dense(units=1, activation=activation)(x_input)

        return keras.Model(inputs=x_input, outputs=output)

    def medium_layer(
        use_wgan: bool = False,
        use_batch_norm: bool = False,
        use_leaky_relu: bool = False,
    ):
        x_input = __dense_layer(
            inputs,
            units=48,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=36,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            inputs,
            units=28,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=18,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            inputs,
            units=12,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=7,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )

        activation = None if use_wgan else "sigmoid"
        output = layers.Dense(units=1, activation=activation)(x_input)

        return keras.Model(inputs=x_input, outputs=output)

    def smallest_layer(
        use_wgan: bool = False,
        use_batch_norm: bool = False,
        use_leaky_relu: bool = False,
    ):
        x_input = __dense_layer(
            x_input,
            units=56,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            inputs,
            units=28,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        x_input = __dense_layer(
            x_input,
            units=7,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )

        activation = None if use_wgan else "sigmoid"
        output = layers.Dense(units=1, activation=activation)(x_input)

        return keras.Model(inputs=x_input, outputs=output)

    model: Any = None

    if nn_size == "big":
        model = big_layer()
    elif nn_size == "small":
        model = medium_layer()
    elif nn_size == "smallest":
        model = smallest_layer()

    return {f"{nn_size}_discriminator": model}


def cycle_gan_generator(input_shape=(56,), nn_size: str = "big") -> keras.Model:

    inputs = layers.Input(shape=input_shape)

    def big_layer(
        use_dropout: bool = False,
        use_batch_norm: bool = True,
        use_leaky_relu: bool = False,
    ):
        embedding = __dense_layer(use_dropout, use_batch_norm, use_leaky_relu)
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        embedding = __dense_layer(use_dropout, use_batch_norm, use_leaky_relu)

        outputs = layers.Dense(units=input_shape[0], activation=None)(embedding)

        return keras.Model(inputs, outputs)

    def medium_layer(
        use_dropout: bool = False,
        use_batch_norm: bool = True,
        use_leaky_relu: bool = False,
    ):
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )

        outputs = layers.Dense(units=input_shape[0], activation=None)(embedding)

        return keras.Model(inputs, outputs)

    def small_layer(
        use_dropout: bool = False,
        use_batch_norm: bool = True,
        use_leaky_relu: bool = False,
    ):
        embedding = __residual_dense_block(
            embedding,
            units=56,
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
            use_leaky_relu=use_leaky_relu,
        )
        outputs = layers.Dense(units=input_shape[0], activation=None)(embedding)

        return keras.Model(inputs, outputs)

    model: Any = None
    if nn_size == "big":
        model = big_layer()
    elif nn_size == "medium":
        model = medium_layer()
    elif nn_size == "small":
        model = small_layer()
    
    return model

if __name__ == "__main__":
    pass
