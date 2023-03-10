""" main.py """

import argparse
import csv
import datetime
import logging
import os
import random
from typing import Any, Dict, Final, List, Tuple

from deepchem.molnet import load_qm9
from deepchem.models.normalizing_flows import NormalizingFlow, NormalizingFlowModel
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import selfies as sf
import tensorflow as tf
import tensorflow_probability as tfp
from rdkit import Chem
from rdkit.Chem import Descriptors as descriptors
from rdkit.Chem.Draw import MolsToGridImage
from tensorflow import keras
# from torch_geometric.data import DataLoader
from wandb.keras import WandbCallback, WandbModelCheckpoint

import wandb
from src.models.generative.gans.converter import (Descriptors, GraphConverter,
                                                  SmilesConverter)
from src.models.generative.gans.network_utils import (
    build_graph_discriminator, build_graph_generator)
from src.models.generative.gans.wgan import GraphWGAN
from src.models.generative.vaes.converter import (DescriptorsVAE,
                                                  GraphConverterVAE,
                                                  SmilesConverterVAE)
from src.models.generative.vaes.gvae import GraphVAE
from src.models.generative.vaes.network_utils import (build_graph_decoder,
                                                      build_graph_encoder)

tfd = tfp.distributions
tfb = tfp.bijectors


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s - %(levelname)s",
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S %p",
    encoding="UTF-8",
    filemode="w",
)


def load_qm(path: str) -> List[str]:
    """read QM9 csv file and extract SMILES"""

    data: List[str] = []

    with open(path, "r") as csv_file:
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

    data: Dict[str, List] = {"smiles": [], "logp": [], "qed": [], "sas": []}

    with open(path, "r") as csv_file:
        rows = csv.reader(csv_file)
        next(rows)

        for row in rows:
            data["smiles"].append(row[0])
            data["logp"].append(float(row[1]))
            data["qed"].append(float(row[2]))
            data["sas"].append(float(row[3]))

    return data


# def load_csv(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     return df


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
    config = {
        "generator_dense_units": [128, 256, 512],
        "dropout": 0.2,
        "adjacency_shape": (
            Descriptors.BOND_DIM.value,
            Descriptors.NUM_ATOMS.value,
            Descriptors.NUM_ATOMS.value,
        ),
        "feature_shape": (Descriptors.NUM_ATOMS.value, Descriptors.ATOM_DIM.value),
        "epochs": 10,
        "batch_size": 32,
        "latent_dim": 64,
        "gconv_units": [128, 128, 128, 128],
        "discriminator_dense_units": [512, 512],
        "generator_opt": keras.optimizers.Adam(5e-4),
        "discriminator_opt": keras.optimizers.Adam(5e-4),
    }

    return config


def sample(
    model: keras.Model, model_type: str, batch_size: int = 32, latent_dim: int = 64
) -> List:

    # LATENT_DIM: Final[int] = 64
    descriptors = None
    if model_type == "GVAE":
        descriptors = DescriptorsVAE
        smilesconverter = SmilesConverterVAE
    else:
        descriptors = Descriptors
        smilesconverter = SmilesConverter

    latent_space = tf.random.normal((batch_size, latent_dim))
    graph = model(latent_space)
    # obtain one-hot encoded adjacency tensor
    adjacency = tf.argmax(graph[0], axis=1)
    adjacency = tf.one_hot(adjacency, depth=descriptors.BOND_DIM.value, axis=1)
    # Remove potential self-loops from adjacency
    adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
    # obtain one-hot encoded feature tensor
    features = tf.argmax(graph[1], axis=2)
    features = tf.one_hot(features, depth=descriptors.ATOM_DIM.value, axis=2)

    return [
        smilesconverter([adjacency[i].numpy(), features[i].numpy()]).transform() for i in range(batch_size)
    ]


def get_number_aromatic_atoms(molecule) -> int:

    aromatic_atoms = [
        molecule.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in range(molecule.GetNumAtoms())
    ]
    count = []
    for _, value in enumerate(aromatic_atoms):
        if value == True:
            count.append(value)
    sum_count = sum(count)

    return sum_count


def calc_aromatic_descriptors(molecule) -> float:
    aromatic_atoms: int = get_number_aromatic_atoms(molecule)
    num_heavy_atoms: int = descriptors.HeavyAtomCount(molecule)

    aromatic_proportion: float = aromatic_atoms / num_heavy_atoms
    return aromatic_proportion


def get_solubility_parameters(smiles) -> List[float]:

    mol_log_p: float = descriptors.MolLogP(smiles)
    mol_weight: float = descriptors.MolWt(smiles)
    num_rot_bonds: int = descriptors.NumRotatableBonds(smiles)
    aromatic_proportion: float = calc_aromatic_descriptors(smiles)

    row = [mol_log_p, mol_weight, num_rot_bonds, aromatic_proportion]
    return row

def get_normalizing_flow_layer(dim: int):
    base_dist = tfd.MultivariateNormalDiag(loc=np.zeros(dim), scale_diag=np.ones(dim))

    if dim % 2 == 0:
        permutation = tf.cast(np.concatenate((np.arange(dim / 2, dim), np.arange(0, dim / 2))), tf.int32)
    else:
        permutation = tf.cast(np.concatenate((np.arange(dim / 2 + 1, dim), np.arange(0, dim / 2))), tf.int32)
        
    num_layers = 8
    flow_layers = []

    Made = tfb.AutoregressiveNetwork(params=2, hidden_units=[521, 521], activation="relu")

    for _ in range(num_layers):
        flow_layers.append(
            (tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=Made))
        )
        
        permutation = tf.cast(np.random.permutation(np.arange(0, dim)), tf.int32)
        flow_layers.append(tfb.Permute(permutation=permutation))

    nf = NormalizingFlow(base_distribution=base_dist, flow_layers=flow_layers)
    
    return nf

def process_smiles(smiles):
    return sf.encoder(smiles)

def get_selfies_alphabet():
    _, datasets, _ = load_qm9(featurizer="ECFP")
    df = pd.DataFrame(data={"smiles" : datasets[0].ids})
    
    data = df[["smiles"]].sample(2500, random_state=42)
    
    sf.set_semantic_constraints() # reset constraints
    constraints = sf.get_semantic_constraints()
    constraints["?"] = 3

    sf.set_semantic_constraints(constraints)
    constraints
    
    data["selfies"] = data["smiles"].apply(process_smiles)

    data["len"] = data["smiles"].apply(lambda x: len(x))
    data.sort_values(by="len").head()
    
    selfies_list = np.asanyarray(data["selfies"])
    selfies_alphabet = sf.get_alphabet_from_selfies(selfies_list)
    selfies_alphabet.add("[nop]") # Ass the "no operation" symbol as a padding character
    selfies_alphabet.add(".")
    selfies_alphabet = list(sorted(selfies_alphabet))
    
    return selfies_alphabet


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser...")
    parser.add_argument("--wgan", help="train WGAN", action="store_true")
    parser.add_argument("--gvae", help="train VAE", action="store_true")
    parser.add_argument("--name", help="Model name", required=True, default="model")
    parser.add_argument("--sample_wgan", help="sample WGAN", action="store_true")
    parser.add_argument("--sample_gvae", help="sample VAE", action="store_true")
    parser.add_argument(
        "--latent", help="Visualize the GVAE latent space", action="store_true"
    )
    parser.add_argument(
        "--solubility", help="train solubility regression model", action="store_true"
    )
    parser.add_argument("--smiles", help="SMILES string")

    args = parser.parse_args()

    # print("adjacency_tensor.shape =", adjacency_tensors.shape)
    # print("feature_tensor.shape =", features_tensors.shape)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    LATENT_DIM: Final[int] = 64

    if args.wgan:

        # logger = logging.getLogger(name="WGAN-training")

        # logger.info("Loading QM9 training dataset...")
        molecules = load_qm("data/qm9.csv")

        adjacency_tensors = []
        features_tensors = []

        for _, molecule in enumerate(molecules):
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

        # logger.info("Finished loading QM9 dataset...")

        adjacency_tensors = tf.convert_to_tensor(adjacency_tensors)
        features_tensors = tf.convert_to_tensor(features_tensors)

        # logger.info(
        #     f"Adjacency tensors shape: {adjacency_tensors.shape}\
        #     - Feature tensors shape: {features_tensors.shape}"
        # )

        wandb.init(project="GAN-small-molecule-generation")
        wandb.config = wandb_initialization()
        config = wandb.config

        # logger.info(
        #     "Finishing setting up WGAN training parameters and logged them to Weights & Biases"
        # )

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

        # logger.info("Successfully built generator and discriminator models")
        # logger.info(generator.summary())
        # logger.info(discriminator.summary())

        # generator.summary()
        # print("======================================================================================== \n")
        # print("======================================================================================== \n")
        # discriminator.summary()

        wgan = GraphWGAN(discriminator_model=discriminator, generator_model=generator)
        # logger.info("Successfully built GraphWGAN model")

        wgan.compile(
            generator_opt=config["generator_opt"],
            discriminator_opt=config["discriminator_opt"],
        )
        # logger.info("Successfully compiled GraphWGAN model")

        history = wgan.fit(
            [adjacency_tensors, features_tensors],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shuffle=True,
            callbacks=[WandbCallback()],
            use_multiprocessing=True,
        )

        wandb.finish()

        # logger.info(
        #     "Finished training GraphWGAN model and saved training weights to Weights and Biases"
        # )

        path_to_save_model = os.path.join(
            os.getcwd(), f"model/generative/gans/{args.name}/{current_time}"
        )
        os.makedirs(path_to_save_model, exist_ok=True)
        tf.saved_model.save(wgan, path_to_save_model)

        # logger.info("Saved GraphWGAN model")

    elif args.gvae:

        # logger = logging.getLogger(name="GVAE-training")
        data = load_zinc("data/zinc.csv")

        adjacency_tensors = []
        features_tensors = []
        qed_tensors = []

        # logger.info("Preparing Zinc dataset...")
        qm_dataset = random.sample(data["smiles"], 10000)
        for i, molecule in enumerate(qm_dataset):
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

        # logger.info("Finished loading Zinc dataset...")

        adjacency_tensors = tf.convert_to_tensor(adjacency_tensors, dtype="float32")
        features_tensors = tf.convert_to_tensor(features_tensors, dtype="float32")
        qed_tensors = tf.convert_to_tensor(qed_tensors, dtype="float32")

        # logger.info(
        #     f"Adjacency tensors shape: {adjacency_tensors.shape} -\
        #                 Features tensors shape: {features_tensors.shape} -\
        #                 QED tensors shape: {qed_tensors.shape}"
        # )

        wandb.init(
            project="VAE-small-molecule-generation",
            name=f"experiment-2500-{current_time}",
        )
        wandb.config = wandb_initialization()
        config = wandb.config

        # logger.info(
        #     "Finishing setting up GVAE training parameters and logged them to Weights & Biases"
        # )

        optimizer = config["generator_opt"]
        encoder = build_graph_encoder(
            gconv_units=[9],
            adjacency_shape=(
                DescriptorsVAE.BOND_DIM.value,
                DescriptorsVAE.NUM_ATOMS.value,
                DescriptorsVAE.NUM_ATOMS.value,
            ),
            features_shape=(
                DescriptorsVAE.NUM_ATOMS.value,
                DescriptorsVAE.ATOM_DIM.value,
            ),
            latent_dim=config["latent_dim"],
            dense_units=config["generator_dense_units"],
            dropout_rate=0.0,  # config["dropout"]
        )
        decoder = build_graph_decoder(
            dense_units=config["generator_dense_units"],
            droupout_rate=config["dropout"],
            latent_dim=config["latent_dim"],
            adjacency_shape=(
                DescriptorsVAE.BOND_DIM.value,
                DescriptorsVAE.NUM_ATOMS.value,
                DescriptorsVAE.NUM_ATOMS.value,
            ),
            feature_shape=(
                DescriptorsVAE.NUM_ATOMS.value,
                DescriptorsVAE.ATOM_DIM.value,
            ),
        )

        # logger.info("Successfully built encoder and decoder models")
        # logger.info(encoder.summary())
        # logger.info(decoder.summary())

        gvae = GraphVAE(
            encoder=encoder, decoder=decoder, latent_dim=config["latent_dim"]
        )
        # logger.info("Successfully built GraphVAE model")
        gvae.compile(optimizer)
        # logger.info("Successfully compiled GraphVAE model")

        history = gvae.fit(
            [adjacency_tensors, features_tensors, qed_tensors],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shuffle=True,
            callbacks=[WandbCallback()],
            use_multiprocessing=True,
        )

        # logger.info(
        #     "Finished training GraphVAE model and saved training weights to Weights and Biases"
        # )
        path_to_save_model = os.path.join(
            os.getcwd(), f"model/generative/vaes/{args.name}/{current_time}"
        )
        os.makedirs(path_to_save_model, exist_ok=True)
        tf.saved_model.save(gvae, path_to_save_model)

        wandb.finish()
        # logger.info("Saved GraphVAE model")

    elif args.sample_gvae:
        path_to_save_model = os.path.join(
            os.getcwd(), f"model/generative/vaes/{args.name}"
        )
        gvae = tf.saved_model.load(path_to_save_model)

        # molecules = gvae.inference(batch_size = 1000) # Not working
        molecules = sample(model=gvae.decoder, model_type="GVAE")

        smiles = [Chem.MolToSmiles(mol.GetMol()) for mol in molecules if mol]

        imgs = MolsToGridImage(
            [mol for mol in molecules if mol],
            molsPerRow=5,
            subImgSize=(150, 150),
            returnPNG=False,
        )

        imgs.save(os.path.join(os.getcwd(), f"results\\vaes\images\{current_time}.png"))

        with open(
            os.path.join(os.getcwd(), f"results\\vaes\smiles\{current_time}.txt"), "w"
        ) as f:
            for s in smiles:
                f.write(f"{s}\n")

    elif args.sample_wgan:
        path_to_save_model = os.path.join(
            os.getcwd(), f"model/generative/gans/{args.name}"
        )
        wgan = tf.saved_model.load(path_to_save_model)

        molecules = sample(wgan.generator, model_type="WGAN", batch_size=100)

        smiles = [Chem.MolToSmiles(mol.GetMol()) for mol in molecules if mol]

        imgs = MolsToGridImage(
            [mol for mol in molecules if mol],
            molsPerRow=5,
            subImgSize=(150, 150),
            returnPNG=False,
        )

        imgs.save(os.path.join(os.getcwd(), f"results\gans\images\{current_time}.png"))

        with open(
            os.path.join(os.getcwd(), f"results\gans\smiles\{current_time}.txt"), "w"
        ) as f:
            for s in smiles:
                f.write(f"{s}\n")

    elif args.latent:

        data = load_zinc("data/zinc.csv")

        adjacency_tensors = []
        features_tensors = []
        qed_tensors = []

        for i, molecule in enumerate(data["smiles"][:1000]):
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

        adjacency_tensors = tf.convert_to_tensor(adjacency_tensors)
        features_tensors = tf.convert_to_tensor(features_tensors)

        path_to_save_model = os.path.join(os.getcwd(), f"model/generative/vaes/{args.name}")
        gvae = tf.saved_model.load(path_to_save_model)

        z_mean, _ = gvae.encoder([adjacency_tensors, features_tensors])
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=qed_tensors[:1000])
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

    # elif args.prepare_solubility:

    #     tasks: List[str] = ["Solubility"]
    #     loader = dc.data.CSVLoader(
    #         tasks=tasks,
    #         feature_field="SMILES",
    #         featurizer=dc.feat.MolGraphConvFeaturizer(use_edges=True)
    #     )
    #     dataset = loader.create_dataset("data/solubility-dataset-train.csv")

    #     splitter = dc.splits.ButinaSplitter()
    #     train_dataset, valid_dataset, test_dataset =\
    #         splitter.train_valid_test_split(dataset, frac_train=0.7, frac_valid=0.2, frac_test=0.1, seed=42)

    #     with open(file="data/processed/processed-solubility-dataset-train.pkl", mode="wb") as file:
    #         pickle.dump(train_dataset, file=file)
    #     with open(file="data/processed/processed-solubility-dataset-valid.pkl", mode="wb") as file:
    #         pickle.dump(valid_dataset, file=file)
    #     with open(file="data/processed/processed-solubility-dataset-test.pkl", mode="wb") as file:
    #         pickle.dump(test_dataset, file=file)

    elif args.solubility:

        descriptor = Chem.MolFromSmiles(args.smiles)
        predictors = np.array([get_solubility_parameters(descriptor)])

        with open(
            file="model/regressors/solubility-rf-sklearn.model", mode="rb"
        ) as file:
            model = joblib.load(filename=file)

        with open(
            file="model/regressors/solubility-standard-scaler-sklearn.model", mode="rb"
        ) as file:
            scaler = joblib.load(filename=file)

        predicted_solubility = model.predict(scaler.transform(predictors))[0]

        print(
            f"Molecule: {args.smiles}. log(Solubility): {predicted_solubility:.3f} mol/L"
        )

    elif args.sample_flow:
        nf = get_normalizing_flow_layer(dim=2000)
        nfm = NormalizingFlowModel(nf, learning_rate = 1e-4, batch_size = 128, model_dir="model/generative/flow/generative-normalizing-flow")
        nfm.restore()
        
        generated_samples = nfm.flow.sample(20)
        log_probs = nfm.flow.log_probs(generated_samples)
        
        mols = tf.math.floor(generated_samples)
        mols = tf.clip_by_value(mols, 0, 1)
        
        int_to_symbol = dict((i, c) for i, c in enumerate(get_selfies_alphabet()))
        
        mols = mols.numpy().tolist()
        selfies_molecule = sf.encoding_to_selfies(mols, vocab_itos=int_to_symbol, enc_type="one_hot")
        
        smile_molecule = sf.decoder(selfies_molecule)
        
        print("SELFIES: ", selfies_molecule)
        print("SMILES: ", smile_molecule)