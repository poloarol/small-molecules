""" app.py """

import os
from typing import List

import deepchem as dc
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors as descriptors
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from src.models.generative.gans.converter import Descriptors, GraphConverter, SmilesConverter
from src.models.generative.vaes.converter import DescriptorsVAE, GraphConverterVAE, SmilesConverterVAE

main_container = st.container()
generative_container = st.container()

@st.cache(allow_output_mutation=True)
def load_sklearn_models(filename: str):
    path_to_model = os.path.join(os.getcwd(), filename)
    with open(file=path_to_model, mode="rb") as file:
        model = joblib.load(filename=file)
    
    return model

@st.cache(allow_output_mutation=True)
def load_tensorflow_models(filename: str):
    path_to_model = os.path.join(os.getcwd(), filename)
    model = tf.saved_model.load(path_to_model)
    return model

@st.cache(allow_output_mutation=True)
def load_deepchem_models(filename: str):
    pass

def get_number_aromatic_atoms(molecule) -> int:
    
    aromatic_atoms = [molecule.GetAtomWithIdx(idx).GetIsAromatic() for idx in range(molecule.GetNumAtoms())]
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

def get_image(molecule):
    return Draw.MolToImage(molecule)

def sample(model: keras.Model, model_type: str, batch_size: int = 32, latent_dim: int = 64) -> List:
    
    # LATENT_DIM: Final[int] = 64
    descriptors = None
    if model_type == "GVAE":
        descriptors = DescriptorsVAE
        # smiles_converter = SmilesConverterVAE
    else:
        descriptors = Descriptors
        # smiles_converter = SmilesConverter
    
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
        SmilesConverter([adjacency[i].numpy(), features[i].numpy()]).transform() \
            for i in range(batch_size)
    ]


solubility_scaler = load_sklearn_models("model/regressors/solubility-standard-scaler-sklearn.model")
solubility_model = load_sklearn_models("model/regressors/solubility-rf-sklearn.model")

with main_container:
    st.title("Generative modelling of small molecules")
    st.write("In this project we aim to use generative models to design small molecules and predict their solubility")
    st.write("""
             1. Generative models
             - Graph Variational Autoencoder
             - Wasserstein GAN
             - Normalizing Generative Flow
             2. Solubility prediction
             - Random Forest
             """)


with generative_container:
    model = st.radio("Choose a generative model.", ("", "GAN", "VAE", "NF"), horizontal=True)
    # TODO: Add ability to navigate along generated molecules. Add NL model
    if model == "GAN":
        wgan = load_tensorflow_models("model/generative/gans/qm9/2022-12-28_18-18-34")
        molecules = sample(wgan.generator, model_type="WGAN")
        molecules = [molecule for molecule in molecules if molecule]
        
        if molecules:
            smiles = [Chem.MolToSmiles(mol.GetMol()) for mol in molecules if mol]
            
            st.text("Generating small molecules using WGAN...")
            
            img_col, props_col = st.columns(2)
            img = get_image(molecules[0])
            
            img_col.text(f"Generated {len(smiles)} valid molecules... ")
            img_col.image(img, caption=smiles[0])
            
            descriptor = Chem.MolFromSmiles(smiles[0])
            predictors = np.array([get_solubility_parameters(descriptor)])
            predicted_solubility = solubility_model.predict(solubility_scaler.transform(predictors))[0]
            
            props_col.text("Physico-Chemical")
            props_col.text(f"Molecular Weight: {descriptors.MolWt(descriptor):.3f} g/mol")
            props_col.text(f"log Solubility: {predicted_solubility:.3f} mol/L")
        else:
            st.text("No valid molecule was generated. Try Again!")
        
    elif model == "VAE":
        gvae = load_tensorflow_models("model/generative/vaes/zinc/2022-12-28_16-10-54")
        molecules = sample(model=gvae.decoder, model_type="GVAE")
        if molecules:
            molecules = [molecule for molecule in molecules if molecule]
            smiles = [Chem.MolToSmiles(mol.GetMol()) for mol in molecules if mol]
            
            st.text("Generating small molecules using GVAE...")
            
            img_col, props_col = st.columns(2)
            img = get_image(molecules[0])
            
            img_col.text(f"Generated {len(molecules)} valid molecules...")
            img_col.image(img, caption=smiles[0])
            
            descriptor = Chem.MolFromSmiles(smiles[0])
            predictors = np.array([get_solubility_parameters(descriptor)])
            predicted_solubility = solubility_model.predict(solubility_scaler.transform(predictors))[0]
            
            props_col.text("Physico-Chemical")
            props_col.text(f"Molecular Weight: {descriptors.MolWt(descriptor):.3f} g/mol")
            props_col.text(f"log Solubility: {predicted_solubility:.3f} mol/L")
        else:
            st.text("No valid molecule was generated. Try Again!")
        
    else:
        pass