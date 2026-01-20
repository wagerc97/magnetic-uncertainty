# Modelling magnetic material properties with uncertainty-aware neural networks

Supplementary material to the publication "Modelling magnetic material properties with uncertainty-aware neural networks".

This repository contains the source code to define the machine learning models to predict intrinsic magnetic material properties, anisotropy field $\mu_0H_\mathrm{a}$ (T) and spontaneous magnetization $\mu_0M_\mathrm{s}$ (T), using approximate Bayesian neural networks, random forests with bagging and gaussian process regression.

The datasets that were either generated or analyzed for intrinsic material property prediction are considered proprietary information and are the intellectual property of TOYOTA Motor Company. 

The source code and train dataset for the graph neural network models predicting coercivity $\mu_0H_\mathrm{c}$ (T) can be found in the GitHub repository [heisammoustafa/Micromagnetic_GNN](https://github.com/heisammoustafa/Micromagnetic_GNN). 


## Environment 
Install dependencies in a new environment (we recommend micromamba) using 
```bash 
micromamba env create -f environment.yml
```


## Usage 
Use the corresponding jupyter notebook to train and evaluate the models. 
Results will be saved in the `models` folder at the project root under a timestamped directory. 
