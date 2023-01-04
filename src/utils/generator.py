""" generator.py """

from typing import Tuple

import numpy as np
from converter import Descriptors, GraphConverter
# import tensorflow as tf
from tensorflow import keras

# def set_seed(seed: int = 1024) -> None:
#     tf.random.set_seed(seed=seed)
#     np.random.seed(seed=seed)

# set_seed()


class DataGenerator(keras.utils.Sequence):
    """Loads data provide a train and test"""

    def __init__(
        self, data, mapping, max_length: int, batch_size: int = 32, shuffle: bool = True
    ) -> None:
        super().__init__()
        self.index: int
        self.data = data
        self.mapping = mapping
        self.max_length: int = max_length
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle

        self.__on_epoch_end()

    def __on_epoch_end(self) -> None:
        """shuffle dataset"""
        self.index = np.arrange(len(self.data.index.tolist()))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple:
        indices = self.data.index.tolist()
        if (index + 1) * self.batch_size > len(indices):
            self.batch_size = len(indices) - index * self.batch_size
            # Generate a single batch
            # Generate indices of the batch
            idx: int = indices[index * self.batch_size : (index + 1) * self.batch_size]
            # Find list od IDs
            batch = [indices[k] for k in idx]
            mol_features, mol_property = self.data_generation(batch)

            return mol_features, mol_property
        return ()

    def load(self, idx: int) -> Tuple:
        """load observation given an index"""
        qed = self.data.loc[idx]["qed"]
        smiles = self.data.loc[idx]["smiles"]
        adjacency, features = GraphConverter(molecule=smiles).transform()

        return adjacency, features, qed

    def data_generation(self, batch) -> Tuple:
        """given a batch, generate a sample"""
        x_1 = np.empty(
            (self.batch_size, Descriptors.BOND_DIM, self.max_length, self.max_length)
        )
        x_2 = np.empty((self.batch_size, self.max_length, len(self.mapping)))
        x_3 = np.empty((self.batch_size,))

        for i, batch_id in enumerate(batch):
            (
                x_1[
                    i,
                ],
                x_2[
                    i,
                ],
                x_3[
                    i,
                ],
            ) = self.load(batch_id)
            return [x_1, x_2], x_3


if "__main__" == __name__:
    pass
