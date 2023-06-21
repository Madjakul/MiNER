#!/bin/bash

LANG="en"
LABEL_COMPLETION=0
# CORPUS_PATH="./data/wikigold/wiki_train_corpus.txt"
# CONLL_PATH="./data/wikigold/distant/wiki_train.conll"
# GAZETTEERS_PATH="./data/wikigold/gazetteers/"
# UNK_GAZETTEERS_PATH="./data/wikigold/gazetteers/UNK.txt"

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Mining Phrases ===${reset}
python3 preprocess_miner.py \
    --lang ${LANG:-"en"} \
    --label_completion ${LABEL_COMPLETION:-1} \
    --corpus_path ${CORPUS_PATH:-"./data/bc5cdr/cdr_train_corpus.txt"} \
    --conll_path ${CONLL_PATH:-"./data/bc5cdr/distant/cdr_train.conll"} \
    --gazetteers_path ${GAZETTEERS_PATH:-"./data/bc5cdr/gazetteers/"} \
    --unk_gazetteers_path ${UNK_GAZETTEERS_PATH:-"./data/bc5cdr/gazetteers/UNK.txt"}
echo ${green}--- Done ---${reset}
