# IUNet
     
1. [**About**](#about)
2. [**Getting started**](#getting-started)
     1. [**Prerequisites**](#prerequisites)
     2. [**Usage**](#usage)
3. [**Contact**](#contact)


# About
IUNet performs surgical tools detection using Inception architecture hierarchical prediction refinement with causal, dilated convolutions for surgical phase recognition.
     
# Getting started
Follow these steps to get the code running on your local machine!

## Prerequisites

```
pip install -r requirements.txt
```

## Usage

The publicly available [Cholec80 dataset](http://camma.u-strasbg.fr/datasets) is used to train and validate the model.

We are using the publicly available [Cholec80 dataset](http://camma.u-strasbg.fr/datasets).

### Stage 1 - Train Feature Extractor

Run:
```
python train.py -c modules/cnn/config/config_feature_extract.yml
```
This will train the CNN feature extractor and in the *Test Step* it will extract for each Video the features of all images and save it as *.pkl*

### Stage 2 - Train Temporal Convolutional Network

```
python train.py -c modules/mstcn/config/config_tcn.yml
```

# Contact

For any question, send an email to mahmoudfazeli89@gmail.com
