""" main.py """

import argparse
import csv
import datetime
import os
from typing import Dict, Final, List, Tuple, Any

import tensorflow as tf
import wandb
from rdkit import Chem
from tensorflow import keras
from wandb.keras import WandbCallback, WandbModelCheckpoint

from src.models.gans.converter import Descriptors, GraphConverter
from src.models.gans.network_utils import (build_graph_discriminator,
                                           build_graph_generator)
from src.models.gans.wgan import GraphWGAN
from src.models.vaes.gvae import GraphVAE
from src.models.vaes.network_utils import (build_graph_decoder,
                                           build_graph_encoder)

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

def wandb_initialization() -> Dict[str, Any]:
    # Start a run, tracking hyperparameters
    # track hyperparameters and run metadata with wandb.config
    config={
        "generator_dense_units": [128, 256, 512],
        "dropout": 0.2,
        "adjacency_shape": (
            Descriptors.BOND_DIM.value,
            Descriptors.NUM_ATOMS.value,
            Descriptors.NUM_ATOMS.value,
        ),
        "feature_shape": (
            Descriptors.NUM_ATOMS.value, 
            Descriptors.ATOM_DIM.value
        ),
        "epochs": 10,
        "batch_size": 16,
        "latent_dim": 64,
        "gconv_units": [128, 128, 128, 128],
        "discriminator_dense_units": [512, 512],
        "generator_opt": keras.optimizers.Adam(5e-4),
        "discriminator_opt": keras.optimizers.Adam(5e-4)
    }
    
    return config


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
        
        for molecule in molecules[:5000]:
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
        
        wandb.init(project="GAN-small-molecule-generation")
        wandb.config = wandb_initialization()
        config = wandb.config
        
        generator = build_graph_generator(
            dense_units=config["generator_dense_units"],
            droupout_rate=config["dropout"],
            latent_dim=config["latent_dim"],
            adjacency_shape=config["adjacency_shape"],
            feature_shape=config["feature_shape"],
        )
        discriminator = build_graph_discriminator(
            gconv_units=config["gconv_units"],
            dense_units=config["discriminator_dense_units"],
            droupout_rate=config["dropout"],
            adjacency_shape=config["adjacency_shape"],
            feature_shape=config["feature_shape"],
        )

        # generator.summary()
        # print("======================================================================================== \n")
        # print("======================================================================================== \n")
        # discriminator.summary()

        wgan = GraphWGAN(discriminator_model=discriminator, generator_model=generator)

        wgan.compile(
            generator_opt=config["generator_opt"],
            discriminator_opt=config["discriminator_opt"],
        )
        
        # wandb.watch([wgan.generator, wgan.discriminator], log="all")
        
        history = wgan.fit(
            [adjacency_tensors, features_tensors], 
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=[
                WandbCallback()
            ]
        )
        
        wandb.finish()
        
        path_to_save_model = os.path.join(os.getcwd(), "models/gans/graph_wgan")
        os.makedirs(path_to_save_model, exist_ok=True)
        tf.saved_model.save(wgan, path_to_save_model)

    elif args.gvae:
        
        data = load_zinc("data/zinc.csv")
        
        adjacency_tensors = []
        features_tensors = []
        qed_tensors = []
        
        for i, molecule in enumerate(data["smiles"][:2500]):
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
        
        current_time = str(datetime.datetime.now())
        wandb.init(project="VAE-small-molecule-generation", name=f"experiment-2500-{current_time}")
        wandb.config = wandb_initialization()
        config = wandb.config
        
        adjacency_tensors = tf.convert_to_tensor(adjacency_tensors, dtype="float32")
        features_tensors = tf.convert_to_tensor(features_tensors, dtype="float32")
        qed_tensors = tf.convert_to_tensor(qed_tensors, dtype="float32")
                
        optimizer = config["generator_opt"]
        encoder = build_graph_encoder(gconv_units=[9],
                                      adjacency_shape=config["adjacency_shape"],
                                      features_shape=config["feature_shape"],
                                      latent_dim=config["latent_dim"],
                                      dense_units=[512],
                                      dropout_rate=config["dropout"])
        decoder = build_graph_decoder(dense_units=config["generator_dense_units"],
                                      droupout_rate=config["dropout"],
                                      latent_dim=config["latent_dim"],
                                      adjacency_shape=config["adjacency_shape"],
                                      feature_shape=config["feature_shape"]
                                    )
        
        
        
        gvae = GraphVAE(encoder=encoder, decoder=decoder, latent_dim=config["latent_dim"])
        gvae.compile(optimizer)
        
        history = gvae.fit(
            [adjacency_tensors, features_tensors, qed_tensors], 
            epochs=config["epochs"],
            shuffle=True,
            callbacks=[WandbCallback()]
            )
        path_to_save_model = os.path.join(os.getcwd(), "models/vaes/graph_vae_3000.h5")
        os.makedirs(path_to_save_model, exist_ok=True)
        gvae.save(path_to_save_model)
        
        wandb.finish()