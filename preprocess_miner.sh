#!/bin/bash

# LANG="en"
# TRAIN_CORPUS_PATH=""
# VAL_CORPUS_PATH=""
# TRAIN_CONLL_PATH=""
# VAL_CONLL_PATH=""
# GAZETTEERS_PATH=""
# UNK_GAZETTEERS_PATH=""

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Mining Phrases ===${reset}
python3 preprocess_miner.py \
    --lang ${LANG:-"en"} \
    --train_corpus_path ${TRAIN_CORPUS_PATH:-"./data/bc5cdr/cdr_train_corpus.txt"} \
    --val_corpus_path ${VAL_CORPUS_PATH:-"./data/bc5cdr/cdr_val_corpus.txt"} \
    --train_conll_path ${TRAIN_CONLL_PATH:-"./data/bc5cdr/cdr_train.conll"} \
    --val_conll_path ${VAL_CONLL_PATH:-"./data/bc5cdr/cdr_val.conll"} \
    --gazetteers_path ${GAZETTEERS_PATH:-"./data/gazetteers/"} \
    --unk_gazetteers_path ${UNK_GAZETTEERS_PATH:-"./data/gazetteers/UNK.txt"}
echo ${green}--- Done ---${reset}
