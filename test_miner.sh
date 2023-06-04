#!/bin/bash

# LANG="en"
# TEST_CORPUS_PATH=""
# LABELS_PATH=""
# LM_PATH=""
# MAX_LENGTH=0
# NER_BATCH_SIZE=0
# NER_PATH=""

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Testing ===${reset}
python3 test_miner.py \
    --lang ${LANG:-"en"} \
    --test_corpus_path ${TEST_CORPUS_PATH:-"./data/bc5cdr/cdr_test.conll"} \
    --labels_path ${LABELS_PATH:-"./data/labels.txt"} \
    --lm_path ${LM_PATH:-"roberta-base"} \
    --max_length ${MAX_LENGTH:-256} \
    --ner_batch_size ${NER_BATCH_SIZE:-4} \
    --ner_path ${NER_PATH:-"./tmp/ner.pt"}
echo ${green}--- Done ---${reset}
