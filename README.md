# MiNER
---

MiNER: a way to learn an efficient named entity recognition model, using only dictionaries.


## Overview

![](https://drive.google.com/uc?id=1avgJPm7JLxnqA2-a92QLpJq-jOvUBL6P)

MiNER's architecture is threefold: (1) distant annotation, with unsu pervised label completion, (2) a revised fine-tuning on a pre-trained language model with a partial conditional random field (CRF), and (3) label distribution smoothing using self-training.


## Benchmark

During our experiments on biomedical NER, we achieved the best F1 over [existing distantly supervised NER methods](https://github.com/yumeng5/RoSTER), but well below on open-domain NER.

TBD


## Installation

### Dependencies

```sh
numpy
tqdm                            ~=4.65.0
datasets                        ~=2.10.1
sentencepiece                   ~=0.1.97
spacy                           ~=3.5.1
torch                           ~=1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
transformers                    ~=4.27.3
seqeval
wandb
```

### Install

```sh
pip install -r requirements.txt
```

## Quickstart

### Running with Provided Datasets

The default dataset to run is the BioCreative V, just run those command one after the others to train and test on this dataset.

```sh
./scripts/pretrain_miner.sh
```

```sh
./scripts/train_partial_ner.sh
```

```sh
./scripts/test_partial_ner.sh
```

```sh
./scripts/train_smooth_ner.sh
```

```sh
./scripts/test_smooth_ner.sh
```

If you want to use an other one of the provided datasets, go into the `scripts` folder and change the necessary pathes to the ones needed.


### Running with Custom Datasets

You'll need:
* A raw `txt` file where each line represent a text.
* A `txt` file listing all possible labels in IOB format, starting by "O".
* `txt` dictionnaries organised the same way as for the other datasets (e.g., one ditionary = one entity type with an explicit name, and one gazetteer per line).
* `txt` dictionary anme "UNK.txt" with a list of your own quality potential phrases from the raw `txt` file. You could use [AutoNER](https://github.com/shangjingbo1226/AutoNER) to get them for example.

Once you have all the necessary documents, go into the scripts folder and change the differents paths accordingly.

You can now run the same commands as shown in the previous part.
