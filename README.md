# HraphNN
Code for "Hybrid region and population hypergraph neural network based on blood oxygenation level-dependent signal for mild cognitive impairment detection"

## Description
![image](https://github.com/lh-wu/HraphNN/edit/main/misc/HraphNN.png)
<p align="center">Fig. 1. Framework of our proposed method</p>

As is shown in Fig.1. , we propose a Hybrid Region and Population Hypergraph Neural Network (HraphNN) for MCI detection. Overall framework of the proposed HraphNN with three steps:(1) Region hypergraph construction conduct selection from the view of BOLD signal and brain region to retain informative signal and brain regions from rs-fMRI data. (2) Population hypergraph construction utilize an improved Large Margin Nerest Neighbor (iLMNN) module to attain the demographic correlation matrix S<sub>demo</sub>, which is integrated with imaging correlation matrix S<sub>img</sub> through multiplying a weighting parameter &lambda; to construct population hypergraph. (3) The proposed population-to-region hypergraph neural network, which conducts convolution from population hypergraph to region hypergraph, mainly consists of population message passing layer (PMP), region convolution layer (RC) and multi-layer perception (MLP), which is utilized to integrate the topological information from the population and brain regions, producing detection results for each subject.

## Requirements
python==3.8

torchvision=0.13.1

dhg==0.9.3

Dependency packages can be installed using following command:

pip install -r requirements.txt

## Dateset

We use three publicly available datasets, including ADNI2、ADNI3(http://adni.loni.usc.edu/) and ABIDE(http://preprocessed-connectomes-project.org/abide/) for experiments.


### Data preprocess
rs-fMRI: standard preprocess protocol with DPABI toolbox(http://rfmri.org/)

To be specific, the first ten volumes are discarded to allow for magnetization equilibrium. The remaining volumes are processed by conversion format, slice time correction, head movement correction, registration, segmentation, spatial standardization, smoothing and AAL116 template segmentation.

The configuration example for preprocessing data using DPABI is shown in the file as follows (./misc/config_for_preprocess.mat).

## Run the code

### 1.Train
python train_eval_HraphNN.py --train=1

### 2.Test
python train_eval_HraphNN.py --train=0# HraphNN

