""" main.py """

import argparse
import csv
import datetime
import os
from typing import Dict, Final, List, Tuple, Any

import tensorflow as tf
import wandb
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback, WandbModelCheckpoint

from src.models.gans.converter import Descriptors, GraphConverter, SmilesConverter
from src.models.vaes.converter import DescriptorsVAE, GraphConverterVAE, SmilesConverterVAE
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


def smiles_to_graph_vae(smiles: str) -> Tuple:
    graph_converter = GraphConverterVAE(smiles)
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
        "batch_size": 32,
        "latent_dim": 64,
        "gconv_units": [128, 128, 128, 128],
        "discriminator_dense_units": [512, 512],
        "generator_opt": keras.optimizers.Adam(5e-4),
        "discriminator_opt": keras.optimizers.Adam(5e-4)
    }
    
    return config

def sample(model: keras.Model, batch_size: int = 32) -> List:
    LATENT_DIM: Final[int] = 64
    latent_space = tf.random.normal((batch_size, LATENT_DIM))
    graph = model(latent_space)
    # obtain one-hot encoded adjacency tensor
    adjacency = tf.argmax(graph[0], axis=1)
    adjacency = tf.one_hot(adjacency, depth=Descriptors.BOND_DIM.value, axis=1)
    # Remove potential self-loops from adjacency
    adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
    # obtain one-hot encoded feature tensor
    features = tf.argmax(graph[1], axis=2)
    features = tf.one_hot(features, depth=Descriptors.ATOM_DIM.value, axis=2)
        
    return [
        SmilesConverter([adjacency[i].numpy(), features[i].numpy()]).transform() \
            for i in range(batch_size)
    ]


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser("Argument parser...")
    parser.add_argument("--wgan", help="train WGAN", action="store_true")
    parser.add_argument("--gvae", help="train VAE", action="store_true")
    parser.add_argument("--name", help="Model name", required=True, default="model")
    parser.add_argument("--sample_wgan", help="sample WGAN", action="store_true")
    parser.add_argument("--sample_gvae", help="sample VAE", action="store_true")
    
    args = parser.parse_args()
    
    # print("adjacency_tensor.shape =", adjacency_tensors.shape)
    # print("feature_tensor.shape =", features_tensors.shape)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    LATENT_DIM: Final[int] = 64
    
    if args.wgan:
        
        molecules = load_qm("data/qm9.csv")

        adjacency_tensors = []
        features_tensors = []
        
        for molecule in molecules:
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
                
        history = wgan.fit(
            [adjacency_tensors, features_tensors], 
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shuffle=True,
            callbacks=[
                WandbCallback()
            ],
            use_multiprocessing=True
        )
        
        wandb.finish()
        
        path_to_save_model = os.path.join(os.getcwd(), f"models/gans/{args.name}/{current_time}")
        os.makedirs(path_to_save_model, exist_ok=True)
        tf.saved_model.save(wgan, path_to_save_model)

    elif args.gvae:
        
        data = load_zinc("data/zinc.csv")
        
        adjacency_tensors = []
        features_tensors = []
        qed_tensors = []
        
        for i, molecule in enumerate(data["smiles"][:2000]):
            smiles = None
            try:
                smiles = Chem.MolFromSmiles(molecule)
            except Exception as err:
                # raise sf.EncoderError("Error during encoding")
                # selfie = Chem.MolFromSmiles(molecule)
                pass
            else:
                pass

            adjacency, features = smiles_to_graph_vae(smiles)
            adjacency_tensors.append(adjacency)
            features_tensors.append(features)
            qed_tensors.append(data["qed"][i])
        
        # wandb.init(project="VAE-small-molecule-generation", name=f"experiment-2500-{current_time}")
        # wandb.config = wandb_initialization()
        config = wandb_initialization()
        
        adjacency_tensors = tf.convert_to_tensor(adjacency_tensors, dtype="float32")
        features_tensors = tf.convert_to_tensor(features_tensors, dtype="float32")
        qed_tensors = tf.convert_to_tensor(qed_tensors, dtype="float32")
                
        optimizer = config["generator_opt"]
        encoder = build_graph_encoder(gconv_units=[9],
                                      adjacency_shape=(
                                          DescriptorsVAE.BOND_DIM.value,
                                          DescriptorsVAE.NUM_ATOMS.value,
                                          DescriptorsVAE.NUM_ATOMS.value
                                        ),
                                      features_shape=(
                                          DescriptorsVAE.NUM_ATOMS.value,
                                          DescriptorsVAE.ATOM_DIM.value
                                        ),
                                      latent_dim=435, #config["latent_dim"],
                                      dense_units=[512],
                                      dropout_rate=config["dropout"])
        decoder = build_graph_decoder(dense_units=config["generator_dense_units"],
                                      droupout_rate=config["dropout"],
                                      latent_dim=435, #config["latent_dim"],
                                      adjacency_shape=(
                                          DescriptorsVAE.BOND_DIM.value,
                                          DescriptorsVAE.NUM_ATOMS.value,
                                          DescriptorsVAE.NUM_ATOMS.value
                                        ),
                                      feature_shape=(
                                          DescriptorsVAE.NUM_ATOMS.value,
                                          DescriptorsVAE.ATOM_DIM.value
                                        )
                                    )
        
        
        
        gvae = GraphVAE(encoder=encoder, decoder=decoder, latent_dim=config["latent_dim"])
        gvae.compile(optimizer)
                
        history = gvae.fit(
            [adjacency_tensors, features_tensors, qed_tensors], 
            epochs=config["epochs"],
            shuffle=True,
            callbacks=[
                # WandbCallback()
                ],
            use_multiprocessing=True
            )
        path_to_save_model = os.path.join(os.getcwd(), f"models/vaes/{args.name}/{current_time}")
        os.makedirs(path_to_save_model, exist_ok=True)
        tf.saved_model.save(gvae, path_to_save_model)
        
        # wandb.finish()
    
    elif args.sample_gvae:
        path_to_save_model = os.path.join(os.getcwd(), f"models/vaes/{args.name}")
        gvae = tf.saved_model.load(path_to_save_model)
        
        molecules = sample(gvae.decoder)
        
    elif args.sample_wgan:
        path_to_save_model = os.path.join(os.getcwd(), f"models/gans/{args.name}")
        wgan = tf.saved_model.load(path_to_save_model)
        
        model = wgan.generator
        
        molecules = sample(model)
        
        smiles = [Chem.MolToSmiles(mol.GetMol()) for mol in molecules if mol]
        
        imgs = MolsToGridImage(
                    [mol for mol in molecules if mol], molsPerRow=5, subImgSize=(150, 150), returnPNG=False
                )
        
        imgs.save(os.path.join(os.getcwd(), f"results\gans\images\{current_time}.png"))
        
        with open(os.path.join(os.getcwd(), f"results\gans\smiles\{current_time}.txt"), "w") as f:
            for s in smiles:
                f.write(f"{s}\n")