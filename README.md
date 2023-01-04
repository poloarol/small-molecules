# Small Molecule

Motivation:
-----------
The development of new drugs (molecules) can be extremely time-consuming and costly.
The use of deep learning models can alleviate the search for good candidate drugs,
by predicting properties of known molecules (e.g., solubility, toxicity, affinity to target protein, etc.).

As the number of possible molecules is astronomical, the space in which we search
for/explore molecules is just a fraction of the entire space. Therefore, it's arguably
desirable to implement generative models that can learn to generate novel molecules
(which would otherwise have never been explored).

SMILES expresses the structure of a given molecule in the form of an ASCII string.
The SMILES string is a compact encoding which, for smaller molecules, is relatively human-readable.
Encoding molecules as a string both alleviates and facilitates database and/or web
searching of a given molecule. RDKit uses algorithms to accurately transform a given
SMILES to a molecule object, which can then be used to compute a great number of
molecular properties/features.

## Generative Modelling

Generative modelling has the potential of uncovering novel therapeutics
that modulate targets and thereby affect the downwstream metabolism.

## Solubility Prediction

Aqueous solubility is a key factor in drug discovery, since if a molecule is not soluble. it will typically be poorly bioavailable, making it difficult to perform <i>in-vivo</i> studies with it, and hence deliver to patients.

Dataset
-------

## Generative Modelling

The dataset used in this tutorial is a quantum mechanics dataset (QM9), obtained from MoleculeNet. Although many feature and label columns come with the dataset, we'll only focus on the SMILES column.

QM9 dataset is a good first dataset to work with for generating graphs, as the maximum number
of heavy (non-hydrogen) atoms found in a molecule is only nine.

1. <b>QM9: </b> https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv

2. <b>Zink: </b> https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv

## Solubility Prediction

1. [ESOL](https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt)
2. [AqSolDB](https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset)
3. [DSL-100] (https://risweb.st-andrews.ac.uk/portal/en/datasets/dls100-solubility-dataset(3a3a5abc-8458-4924-8e6c-b804347605e8).html)

Feature Engineering
-------------------

## Generative Modelling

<b>Representing a molecular graph</b>. Molecules can naturally be expressed
as undirected graphs G = (V, E), where V is a set of vertices (atoms), and 
a set of edges (bonds). As for this implementation, each graph (molecule) will
be represented as an adjacency tensor A, which encodes existence/non-existence of
atom-pairs with their one-hot encoded bond types stretching an extra dimension,
and a feature tensor H, which for each atom, one-hot encodes its atom type. Notice,
as hydrogen atoms can be inferred by RDKit, hydrogen atoms are excluded from A and H for
easier modeling.

## Solubility Prediction

We use easy to calculate RDKit descriptors, as described by [<i> Yalkowsky et al. </i>](https://jpharmsci.org/article/S0022-3549(16)30715-8/fulltext). These descriptors are:

1. Octanol-Water partition coefficient
2. Molecular Weight
3. Num. of Rotatable Bonds
4. Aromatic Proportion = Num. Aromatic Atoms / Num Heavy Atoms

## How to Use
-------------
- Create your virtual environment
- Install libraries in the `requirements.txt`
- Getting models `unzip models.zip`

1. Train Models:
    - WGAN-GP: python main.py --wgan --name `name of model`
    - gVAE: python main.py --gvae --name `name of model`
2. Sample latent space:
    - WGAN-GP: python main.py --sample_wgan --name `name of model`
    - gVAE: python main.py --sample_gvae --name `name of model`
3. Visualize GVAE latent space
    - python main.py --latent --name `name of model`
4. Predict solubility
    - python main.py --solubility --name `name of model` --smiles `smiles string`

## Launch Streamlit App locally
-------------------------------
`streamlit run app.py`


Implemented Models
---------------
1. [WGAN-GP with R-GCN for the generation of small molecules graphs](https://keras.io/examples/generative/wgan-graphs/)
2. Relational Graph Convolutional Neural Network Variational AutoEncoder
3. Mol-CycleGAN (Currently implementing)
4. Random Forest solubility predictor


Source:
-------
1. [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973)

2. [GraphVAE: Towards Generation of Small Graphs Using Variational AutoEncoders](https://arxiv.org/pdf/1802.03480.pdf)

3. [Mol-CycleGAN: A generative model for molecular optimization](https://arxiv.org/pdf/1802.03480.pdf)

4. [Junction Tree Variational AutoEncoders for Molecular Graph Generation](https://arxiv.org/abs/1802.04364)

5. [ESOL: Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x)

6. [AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds](https://www.nature.com/articles/s41597-019-0151-1)

7. [Is Experimental Data Quality the Limiting Factor in Predicting the Aqueous Solubility of Driglike Molecules](https://pubs.acs.org/doi/full/10.1021/mp500103r)

6. [FastFlows: Flow-Based Models for Molecular Graph Generation](https://arxiv.org/abs/2201.12419)

Objectives
----------
1. Implement Generative models (VAE and GAN) for small molecule design
2. Add support for the latest diffusion models
3. Sample generative models latent space
4. Provide an API to use models
5. Add a solubility and toxicity prediction models


Conclusion:
-----------
What we've learned, and prospects. In this tutorial, a generative model
for molecular graphs was succesfully implemented, which allowed us to generate novel molecules.

In the future, it would be interesting to implement generative models
that can modify existing molecules (for instance, to optimize solubility or protein-binding of an existing molecule).
For that however, a reconstruction loss would likely be needed, which is
tricky to implement as there's no easy and obvious way to compute similarity
between two molecular graphs.
