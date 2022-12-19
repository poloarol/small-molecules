""" mol_cyclegan.py """


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gan import GAN


class CycleGAN(GAN):
    """ Molecular CycleGAN, for molecular generation.

    Args:
        GAN (Object): Abstract class to build GAN models.
    """