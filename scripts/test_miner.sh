#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
# TEST_CORPUS_PATH="./data/wikigold/gold/wiki_test.conll"
# LABELS_PATH="./data/wikigold/labels.txt"
# LM_PATH="./tmp/wiki_lm"
# MAX_LENGTH=512
# NER_BATCH_SIZE=0
NER_PATH="./tmp/cdr_ner.pt"
# CORRECTED_LOSS=
# GAMMA=

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Testing ===${reset}
python3 test_miner.py \
    --lang ${LANG:-"en"} \
    --test_corpus_path ${TEST_CORPUS_PATH:-"./data/bc5cdr/gold/cdr_test.conll"} \
    --labels_path ${LABELS_PATH:-"./data/bc5cdr/labels.txt"} \
    --lm_path ${LM_PATH:-"roberta-base"} \
    --max_length ${MAX_LENGTH:-256} \
    --ner_batch_size ${NER_BATCH_SIZE:-32} \
    --ner_path ${NER_PATH:-"./tmp/cdr_ner.pt"} \
    --corrected_loss ${CORRECTED_LOSS:-0} \
    --gamma ${GAMMA:-1.0}
echo ${green}--- Done ---${reset}

