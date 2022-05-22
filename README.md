# RNA Secondary Structure Model (RSSM)

This repository hosts a novel RNA secondary structure prediction tool described in the paper: **Integrated pretraining with evolutionary information to improve RNA secondary structure prediction**.

RSSM is a hybrid convolutional and self-attention based solution that predicts RNA secondary structure either in single sequence based or multiple sequence alignment based fashions.

![RSSM model](figures/model_arch.jpg)

Our RSSM models are pretrained with Rfam sequences and/or Rfam deposited MSAs, in a supervised manner by leveraging readily available computational structures as proxy for true RNA secondary structures, i.e. LinearPartition contact map probabilities. Our models are further finetuned in bprna dataset.

We release three types of RSSM models:
- Single sequence based RSSM, equivalent to the $RSSM_{d64,mix}$ ensemble model from the paper 
- Covariance feature based RSSM, equivalent to the $RSSM_{d64,T,rnafold}$ ensemble model from the paper 
- Alignment based RSSM, equivalent to $MSA-RSSM_{d64,T,rnafold}$ in the paper 

## installation and dependency

RSSM is dependent on the following python packages that can be installed on Anaconda
- PyTorch (GPU enabled version)
- numpy
- scipy
- scikit-learn
- pandas
- h5py
- yaml
- tqdm

We strongly suggest interested user installing these aforementioned python packages in a [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment. Then, install the RSSM package locally in the created miniconda environment.

```
git clone https://github.com/HarveyYan/RSSMFold
cd RSSMFold
pip install -e .
```

## single sequence based RSSM

Example usage:
```
RSSMFold --input_fasta_path=examples/input.fasta --out_dir=./output --use_gpu_device 0 1
```

An RSSM model will be loaded onto GPU devices with id 0 and 1. Pretrained weights will be automatically downloaded from a shared dropbox link, upon first ```RSSMFold``` launch. RSSM will then proceed to predict RNA secondary structures for each sequence in ```examples/input.fasta```. Results will be saved to ```./output``` in ```bpseq``` format. In the following text, we will describe some useful command line flags. For further information, please refer to  ```RSSMFold -h```.

- ```--generate_dot_bracket=True``` generates pseudoknot-free RNA secondary structures in dot-bracket format, on top of RSSM predicted structures in ```bpseq``` files. The elimination of pseudoknots is done by the ```FreeKnot``` program, which is included in the RSSM distribution.
- ```--save_contact_map_prob=True``` saves copies of predicted RNA contact map probabilities in ```npy``` format.
- ```--matrix_sampling_mode``` selects an RNA contact map discretization method. The figure below compares a selection of popular sampling techniques. For more details, please refer to our paper.


<p align="center">
  <img src="figures/sampling_comparison.jpg" width="300" />
</p>

- ```--use_lp_pred=True``` augments/ensembles RSSM predicted RNA contact maps with LinearPartition probabilities, which are beneficial for predicting long range basepairs.

```--use_lp_pred=True``` further requires the ```LinearPartition``` software. Please follow the instruction in this [link](https://github.com/LinearFold/LinearPartition) and install ```LinearPartition``` binary at RSSM root directory ```LinearPartition/linearpartition```.

```
git clone https://github.com/LinearFold/LinearPartition
cd LinearPartition
make
cd ..
```

- ```--enable_sliding_window=True``` only predicts RNA basepairs that reside in overlapping sliding windows, which is similar to ```RNAplfold```, an earlier thermodynamic model. The sliding window method trades the ability of predicting potential long range basepairs to higher accuracy at local RNA structures, and enables fitting much longer sequences into GPUs. Its benefits are clearly identified and measured in the long RNA sequence regime, using sequences between 2000-4000 nts in the bprna-1m dataset, as shown in the figure below. For more information please refer to our paper.

![sliding window](figures/sliding_window.jpg)








