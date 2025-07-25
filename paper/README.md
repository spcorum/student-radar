# Note
This file is pulled from the orginal paper git repo with a few modification such that the packages is compatible with the most upd to date package versions. A few large files (e.g. dataset, or checkpoint model) is removed from the code to avoid blow up the storage. 

Before running, the following set up should be added:
The actual dataset should be downloaded from [kaggle](https://www.kaggle.com/datasets/eduardocandioto/raw-radar-data?select=EXP_17_M.h5) and add to the `data/real` folder. Another `preprocessing` folding should be added under the data folder to avoid crashes.

# Raw Radar Data Generation
---

[![arXiv](https://img.shields.io/badge/arXiv%20papr-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2308.02632 
)&nbsp;

This repository is the the code used in the paper: Generation of Realistic Synthetic Raw Radar Data for Automated Driving Applications using Generative Adversarial Networks. Above there a summary of this method.

![Summary of the Approach](summary.png "Summary of the Approach")

## Instalation
---
```console
$ git clone <repository>
```

## Accessing the Dataset

The dataset file `EXP_17_M.h5` is also avaiable on [kaggle](https://www.kaggle.com/datasets/eduardocandioto/raw-radar-data?select=EXP_17_M.h5).

## Usage
---


### Preprocessing 
In order to perform the training is necessary preprocess the data, that are in h5 file. For this run all the jupyter netebook:
```console
save_prerpocessed_chirps_labels.ipynb
```
This notebook will parse the data to numpy array and apply the change of scales. At the end of the execution will have one `.npy` file with the chirps data and other with the labels.

### Install tensorflow-addons and wandb
Run this to install tensorflow-addons, which as the SpectralNorm layer.
```console
pip install tensorflow-addons
pip install wandb
```

### Log into wandb
Log into wandb account. API key is 85019a9c3b05de9fa0211a19fd654750ad845f1f
```console
wandb login
```

Run this to install tensorflow-addons, which as the SpectralNorm layer.
```console
pip install tensorflow-addons
pip install wandb
```

### Training the model
Use the follow script to train the student model.
```console
!python src/models/trainings/train_conditional.py --model_type student
```
And for the original model
Use the follow script to train the student model.
```console
!python src/models/trainings/train_conditional.py --model_type original
```
This code uses the Weights & Bias API.

### Saving the generations
To save the generations in a numpy file.
```console
python src/generate/save_conditional.py
```

### Saving RA maps
To save RA maps as png files.
```console
python src/generate/save_ramaps.py
``
