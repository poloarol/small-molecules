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

https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv

<b>Representing a molecular graph</b>. Molecules can naturally be expressed
as undirected graphs G = (V, E), where V is a set of vertices (atoms), and 
a set of edges (bonds). As for this implementation, each graph (molecule) will
be represented as an adjacency tensor A, which encodes existence/non-existence of
atom-pairs with their one-hot encoded bond types stretching an extra dimension,
and a feature tensor H, which for each atom, one-hot encodes its atom type. Notice,
as hydrogen atoms can be inferred by RDKit, hydrogen atoms are excluded from A and H for
easier modeling.

Models to implement
---------------
1. [WGAN-GP with R-GCN for the generation of small molecules graphs](https://keras.io/examples/generative/wgan-graphs/) (Current implementing)
2. Graph Variational AutoEncoder (Currently implementing)
3. Mol-CycleGAN (Currently implementing)
4. JT-VAE (To be implemented)

To-Do
-----
1. Complete the WGAN model and train it
2. Add support for weights and biases
3. Add better experimental logging
4. Finish implementation of GVAE and JTVAE
5. Finish implementation of mol-cycle-gan

How to Use
----------
TBD

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
