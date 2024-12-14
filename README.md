# Vignette-gnn
Graph Neural Network in Python and Application to Molecule Solubility Prediction (Graph Level Prediction)

### Contributors
- Jaxon Zhang
- Ruizhe Jiang
- Pramukh Shankar

### Abstract
This project aims to implement a **Graph Neural Network (GNN)** in Python and apply it to the prediction of molecule solubility. The dataset used is the [ESOL](https://arxiv.org/abs/1703.00564) dataset that contains 1128 molecules and their solubility values, which is included in the `MoleculeNet` module of `torch_geometric`. A simple GNN is trained on the ESOL dataset and the performance of the GNN is evaluated using the mean squared error (MSE) metric. The GNN is able to achieve a MSE of **xxx** on the ESOL dataset, which is comparable to the performance of other machine learning models on the same dataset.

### Repository Contents

data/ - Folder containing the ESOL dataset; it is also included in the `MoleculeNet` module of `torch_geometric`
img/ - Folder containing images used in the vignette
  img/slides/ - Folder containing images used in the vignette
scripts/ - Folder containing the Python scripts used to implement the GNN and train it on the ESOL dataset
  scripts/draft.ipynb - Jupyter notebook containing the draft code for the GNN implementation
  scripts/model.py - Python script containing the GNN model implementation
  scripts/training.py - Python script containing the training code for the GNN
  scripts/visualization.py - Python script containing the code to visualize the GNN predictions


### Reference List
[A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
