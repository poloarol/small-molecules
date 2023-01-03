""" evaluation.py """


from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.DataStructs import FingerprintSimilarity


def tanimoto_similarity(database_molecules, query_mol):
    
    fingerprints = [FingerprintMol(molecule) for molecule in database_molecules]
    query = FingerprintMol(query_mol)
    
    similarities = []
    
    for idx, fingerprint in enumerate(fingerprints):
        similarities.append((idx, FingerprintSimilarity(query, fingerprint)))
    
    similarities.sort(key = lambda x: x, reverse = True)
    
    return similarities