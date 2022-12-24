# Small Molecule Generative modelling

Generative modelling has the potential of uncovering novel therapeutics
that modulate targets and thereby affect the downwstream metabolism.

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

Dataset
-------
The dataset used in this tutorial is a quantum mechanics dataset (QM9), obtained from MoleculeNet.
Although many feature and label columns come with the dataset, we'll only focus on the SMILES column. 
QM9 dataset is a good first dataset to work with for generating graphs, as the maximum number
of heavy (non-hydrogen) atoms found in a molecule is only nine.

<b>QM9: </b> https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv

<b>Zink: </b> https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv

<b>Representing a molecular graph</b>. Molecules can naturally be expressed
as undirected graphs G = (V, E), where V is a set of vertices (atoms), and 
a set of edges (bonds). As for this implementation, each graph (molecule) will
be represented as an adjacency tensor A, which encodes existence/non-existence of
atom-pairs with their one-hot encoded bond types stretching an extra dimension,
and a feature tensor H, which for each atom, one-hot encodes its atom type. Notice,
as hydrogen atoms can be inferred by RDKit, hydrogen atoms are excluded from A and H for
easier modeling.

Implemented Models
---------------
1. [WGAN-GP with R-GCN for the generation of small molecules graphs](https://keras.io/examples/generative/wgan-graphs/)
2. Relational Graph Convolutional Neural Network Variational AutoEncoder
3. Mol-CycleGAN (Currently implementing)

How to Use
----------

- Create your virtual environment
- Install libraries in the `requirements.txt`

1. Train Models:
    - WGAN-GP: python main.py --wgan --name `name of model`
    - gVAE: python main.py --gvae --name `name of model`
2. Sample latent space:
    - WGAN-GP: python main.py --sample_wgan --name `name of sample`
    - gVAE: python main.py --sample_gvae --name `name of sample`

Objectives
----------
1. Implement Generative models (VAE and GAN) for small molecule design
2. Add support for the latest diffusion models
3. Sample generative models latent space
4. Provide an API to use models

Source:
-------
[MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973)

[GraphVAE: Towards Generation of Small Graphs Using Variational AutoEncoders](https://arxiv.org/pdf/1802.03480.pdf)

[Mol-CycleGAN: A generative model for molecular optimization](https://arxiv.org/pdf/1802.03480.pdf)

[Junction Tree Variational AutoEncoders for Molecular Graph Generation](https://arxiv.org/abs/1802.04364)


Conclusion:
-----------
What we've learned, and prospects. In this tutorial, a generative model
for molecular graphs was succesfully implemented, which allowed us to generate novel molecules.

In the future, it would be interesting to implement generative models
that can modify existing molecules (for instance, to optimize solubility or protein-binding of an existing molecule).
For that however, a reconstruction loss would likely be needed, which is
tricky to implement as there's no easy and obvious way to compute similarity
between two molecular graphs.
