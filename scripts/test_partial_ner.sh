#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
TEST_CORPUS_PATH="$DATA_ROOT/wikigold/gold/wiki_test.conll"
LABELS_PATH="$DATA_ROOT/wikigold/labels.txt"
LM_PATH="$PROJECT_ROOT/tmp/wiki_lm-128"
MAX_LENGTH=256
NER_PATH="$PROJECT_ROOT/tmp/wiki_ner-128.pt"

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Testing ===${reset}
python3 test_partial_ner.py \
    --lang ${LANG:-"en"} \
    --test_corpus_path ${TEST_CORPUS_PATH:-"$DATA_ROOT/bc5cdr/gold/cdr_test.conll"} \
    --labels_path ${LABELS_PATH:-"$DATA_ROOT/bc5cdr/labels.txt"} \
    --lm_path ${LM_PATH:-"$PROJECT_ROOT/tmp/cdr_lm-512"} \
    --max_length ${MAX_LENGTH:-512} \
    --ner_path ${NER_PATH:-"$PROJECT_ROOT/tmp/cdr_ner-512.pt"} 
echo ${green}--- Done ---${reset}

