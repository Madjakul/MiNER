#!/bin/bash

# LANG="en"
# CORPUS_PATH=""
# CONLL_PATH=""
# GAZETTEERS_PATH=""
# UNK_GAZETTEERS_PATH=""

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Mining Phrases ===${reset}
python3 preprocess_miner.py \
    --lang ${LANG:-"en"} \
    --corpus_path ${CORPUS_PATH:-"./data/bc5cdr/cdr_train_corpus.txt"} \
    --conll_path ${CONLL_PATH:-"./data/bc5cdr/distant/cdr_train.conll"} \
    --gazetteers_path ${GAZETTEERS_PATH:-"./data/gazetteers/"} \
    --unk_gazetteers_path ${UNK_GAZETTEERS_PATH:-"./data/gazetteers/UNK.txt"}
echo ${green}--- Done ---${reset}
