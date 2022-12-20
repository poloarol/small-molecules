""" main.py """

import argparse
import csv
from typing import List, Tuple, Final, Dict

from src.models.gans.converter import GraphConverter, Descriptors
from src.models.gans.network_utils import build_graph_discriminator, build_graph_generator
from src.models.gans.wgan import GraphWGAN

from src.models.vaes.gvae import GraphVAE
from src.models.vaes.network_utils import build_graph_encoder, build_graph_decoder

import tensorflow as tf
from tensorflow import keras
from rdkit import Chem


def load_qm(path: str) -> List[str]:
    """ read QM9 csv file and extract SMILES """
    
    data: List[str] = []
    
    with open(path, 'r') as csv_file:
        rows = csv.reader(csv_file)
        next(rows)
        
        for row in rows:
            data.append(row[1])

    return data

def load_zinc(path: str) -> Dict[str, list]:
    """ 
    read zinc csv file and extract SMILES
    and with properties
    """
    
    data: Dict[str, List] = {"smiles": [],
                             "logp": [],
                             "qed": [],
                             "sas": []}
    
    with open(path, 'r') as csv_file:
        rows = csv.reader(csv_file)
        next(rows)

        for row in rows:
            data["smiles"].append(row[0])
            data["logp"].append(float(row[1]))
            data["qed"].append(float(row[2]))
            data["sas"].append(float(row[3]))
    
    return data

def smiles_to_graph(smiles: str) -> Tuple:
    graph_converter = GraphConverter(smiles)
    adjacency, features = graph_converter.transform()
    
    return adjacency, features

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser("Argument parser...")
    parser.add_argument("--wgan", help="train WGAN")
    parser.add_argument("--gvae", help="train VAE")
    parser.add_argument("--tj_vae", help="train JT-VAE")
    
    args = parser.parse_args()
    
    # print("adjacency_tensor.shape =", adjacency_tensors.shape)
    # print("feature_tensor.shape =", features_tensors.shape)
    
    LATENT_DIM: Final[int] = 64
    
    if args.wgan:
        
        molecules = load_qm("data/qm9.csv")
    
        adjacency_tensors = []
        features_tensors = []
        
        for molecule in molecules[:100]:
            smiles = None
            try:
                smiles = Chem.MolFromSmiles(molecule)
            except Exception as err:
                # raise sf.EncoderError("Error during encoding")
                # selfie = Chem.MolFromSmiles(molecule)
                pass
            else:
                pass

            adjacency, features = smiles_to_graph(smiles)
            adjacency_tensors.append(adjacency)
            features_tensors.append(features)
        
        adjacency_tensors = tf.convert_to_tensor(adjacency_tensors)
        features_tensors = tf.convert_to_tensor(features_tensors)
        
        generator = build_graph_generator(
            dense_units=[128, 256, 512],
            droupout_rate=0.2,
            latent_dim=LATENT_DIM,
            adjacency_shape=(
                Descriptors.BOND_DIM.value,
                Descriptors.NUM_ATOMS.value,
                Descriptors.NUM_ATOMS.value,
            ),
            feature_shape=(Descriptors.NUM_ATOMS.value, Descriptors.ATOM_DIM.value),
        )
        discriminator = build_graph_discriminator(
            gconv_units=[128, 128, 128, 128],
            dense_units=[512, 512],
            droupout_rate=0.2,
            adjacency_shape=(
                Descriptors.BOND_DIM.value,
                Descriptors.NUM_ATOMS.value,
                Descriptors.NUM_ATOMS.value,
            ),
            feature_shape=(Descriptors.NUM_ATOMS.value, Descriptors.ATOM_DIM.value),
        )

        # generator.summary()
        # print("======================================================================================== \n")
        # print("======================================================================================== \n")
        # discriminator.summary()

        wgan = GraphWGAN(discriminator_model=discriminator, generator_model=generator)

        wgan.compile(
            generator_opt=keras.optimizers.Adam(5e-4),
            discriminator_opt=keras.optimizers.Adam(5e-4),
        )
        
        wgan.fit([adjacency_tensors, features_tensors], epochs=10, batch_size=16)

    elif args.gvae:
        
        data = load_zinc("data/zinc.csv")
        
        adjacency_tensors = []
        features_tensors = []
        qed_tensors = []
        
        for i, molecule in enumerate(data["smiles"][:200]):
            smiles = None
            try:
                smiles = Chem.MolFromSmiles(molecule)
            except Exception as err:
                # raise sf.EncoderError("Error during encoding")
                # selfie = Chem.MolFromSmiles(molecule)
                pass
            else:
                pass

            adjacency, features = smiles_to_graph(smiles)
            adjacency_tensors.append(adjacency)
            features_tensors.append(features)
            qed_tensors.append(data["qed"][i])
        
        adjacency_tensors = tf.convert_to_tensor(adjacency_tensors, dtype="float32")
        features_tensors = tf.convert_to_tensor(features_tensors, dtype="float32")
        qed_tensors = tf.convert_to_tensor(qed_tensors, dtype="float32")
                
        optimizer = keras.optimizers.Adam(learning_rate=5e-4)
        encoder = build_graph_encoder(gconv_units=[9],
                                      adjacency_shape=(Descriptors.BOND_DIM.value, 
                                                       Descriptors.NUM_ATOMS.value,
                                                       Descriptors.NUM_ATOMS.value),
                                      features_shape=(Descriptors.NUM_ATOMS.value,
                                                      Descriptors.ATOM_DIM.value),
                                      latent_dim=LATENT_DIM,
                                      dense_units=[512],
                                      dropout_rate=0.0)
        decoder = build_graph_decoder(dense_units=[128, 256, 512],
                                      droupout_rate=0.2,
                                      latent_dim=LATENT_DIM,
                                      adjacency_shape=(Descriptors.BOND_DIM.value,
                                                       Descriptors.NUM_ATOMS.value,
                                                       Descriptors.NUM_ATOMS.value),
                                      feature_shape=(Descriptors.NUM_ATOMS.value,
                                                     Descriptors.ATOM_DIM.value))
        
        
        
        model = GraphVAE(encoder=encoder, decoder=decoder, max_lenth=42, latent_dim=LATENT_DIM)
        model.compile(optimizer)
        
        model.fit([adjacency_tensors, features_tensors, qed_tensors], epochs=10)
        
    elif args.jt_vae:
        pass