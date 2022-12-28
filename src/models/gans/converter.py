""" converter.py """

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Final, Tuple

import numpy as np
from rdkit import Chem


class Descriptors(Enum):
    """Provide constantsa and molecular descriptors"""

    SMILE_CHARSET: Final[str] = [
        "C",
        "B",
        "F",
        "I",
        "H",
        "O",
        "N",
        "S",
        "P",
        "Cl",
        "Br",
    ]
    NUM_ATOMS: Final[int] = 9 #120
    BOND_DIM: Final[int] = 5
    ATOM_DIM: Final[int] = 5 #len(SMILE_CHARSET)


class DataConverter(ABC):
    """
    Abstract class to provides method for conversion
    of data between smiles and graph representation
    """

    def __init__(self, molecule: str | Tuple) -> None:
        super().__init__()
        self.molecule = molecule
        self.atom_mapping = self.__get_atom_mapping()
        self.bond_mapping = self.__get_bond_mapping()

    @abstractmethod
    def transform(self) -> Tuple | str:
        """transform the data to specified format"""

    def __get_atom_mapping(self) -> Dict[int, str]:
        """map atoms to indices"""
        # smile_to_idx: Dict[str, int] = {
        #     char: idx for idx, char in enumerate(Descriptors.SMILE_CHARSET.value)
        # }
        # idx_to_simle: Dict[int, str] = {
        #     idx: char for idx, char in enumerate(Descriptors.SMILE_CHARSET.value)
        # }

        # smile_to_idx.update(idx_to_simle)
        # return smile_to_idx
        
        atom_mapping = {
            "C": 0,
            0: "C",
            "N": 1,
            1: "N",
            "O": 2,
            2: "O",
            "F": 3,
            3: "F"
        }
        
        return atom_mapping

    def __get_bond_mapping(self) -> int:
        """provides bond mapping"""
        return {
            "SINGLE": 0,
            0: Chem.BondType.SINGLE,
            "DOUBLE": 1,
            1: Chem.BondType.DOUBLE,
            "TRIPLE": 2,
            2: Chem.BondType.TRIPLE,
            "AROMATIC": 3,
            3: Chem.BondType.AROMATIC,
        }


class GraphConverter(DataConverter):
    """
    Utility class to convert smiles to graphs
    """

    def transform(self) -> Tuple:
        """build an adjacency and feature matrix of the molecule"""
        adjacency: np.array = np.zeros(
            (Descriptors.BOND_DIM.value, Descriptors.NUM_ATOMS.value, Descriptors.NUM_ATOMS.value),
            "float32",
        )
        features: np.array = np.zeros(
            (Descriptors.NUM_ATOMS.value, Descriptors.ATOM_DIM.value), "float32"
        )

        # loop over each atom in the molecule
        for _, atom in enumerate(self.molecule.GetAtoms()):
            atom_idx: int = atom.GetIdx()
            atom_type: str = self.atom_mapping[atom.GetSymbol()]
            features[atom_idx] = np.eye(Descriptors.ATOM_DIM.value)[atom_type]

            # loop over neighbours
            for _, neigbour in enumerate(atom.GetNeighbors()):
                neighbour_idx: int = neigbour.GetIdx()
                bond: str = self.molecule.GetBondBetweenAtoms(atom_idx, neighbour_idx)
                bond_type_idx = self.bond_mapping[bond.GetBondType().name]
                adjacency[
                    bond_type_idx, [atom_idx, neighbour_idx], [neighbour_idx, atom_idx]
                ] = 1
        
        # Where no bond, add 1 to last channel (indicating "non-bond")
        # Notice: channels-first
        
        adjacency[-1, np.sum(adjacency, axis=0) == 0] = 1
        
        # Where no atom, add 1 to last column (indicating "non-atom")
        features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1

        return adjacency, features


class SmilesConverter(DataConverter):
    """
    Utility class to convert graphs to smiles
    """

    def transform(self) -> str:
        """generates a Smiles from a graph"""
        molecule = Chem.RWMol()  # Editable molecule
        adjacency, features = self.molecule

        # remove 'no atoms' and atoms with no bonds
        keep_idx = np.where(
            (np.argmax(features, axis=1) != Descriptors.ATOM_DIM.value - 1)
            & (np.sum(adjacency[:-1], axis=(0, 1)) != 0)
        )[0]
        features = features[keep_idx]
        adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

        # add atoms to molecule
        for atom_type_idx in np.argmax(features, axis=1):
            atom = Chem.Atom(self.atom_mapping[atom_type_idx])
            _ = molecule.AddAtom(atom)

        # add bonds between atoms in molecule,
        # based on the upper triangles of the [symmetric] adjacency matrix

        (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(adjacency) == 1)
        for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
            if atom_i == atom_j or bond_ij == Descriptors.BOND_DIM.value - 1:
                continue
            bond_type = self.bond_mapping[bond_ij]
            molecule.AddBond(int(atom_i), int(atom_j), bond_type)

        # Sanitize the molecule
        flag = Chem.SanitizeMol(molecule, catchErrors=True)

        if flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
        return molecule


if "__main__" == __name__:
    pass