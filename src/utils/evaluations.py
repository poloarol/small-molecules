""" evaluation.py """

from typing import List

from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.DataStructs import FingerprintSimilarity


def tanimoto_similarity(database_molecules: List, query_mol) -> List[float]:
    """
    Calculate the Tanimoto similarity of a molecule against a database

    database_molecules: List of rdkit molecules
    query_mol: target molecule to calculate similarity against
    """

    fingerprints = [FingerprintMol(molecule) for molecule in database_molecules]
    query = FingerprintMol(query_mol)

    similarities = []

    for _, fingerprint in enumerate(fingerprints):
        similarities.append(FingerprintSimilarity(query, fingerprint))

    # similarities.sort(key = lambda x: x, reverse = True)

    return similarities
