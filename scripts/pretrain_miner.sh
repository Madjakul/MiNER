#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
TRAIN_CORPUS_PATH="$DATA_ROOT/conll/conll_train_corpus.txt"
VAL_CORPUS_PATH="$DATA_ROOT/conll/conll_test_corpus.txt"
MAX_LENGTH=128
LM_PATH="./tmp/conll_lm-128"
# SEED=0
# MLM_PROBABILITY=0.0
LM_TRAIN_BATCH_SIZE=16
# MAX_STEPS=0
# LM_ACCUMULATION_STEPS=4
MAX_STEPS=5000

WANDB=

# *****************************************************************************


green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

cmd=( python3 pretrain_miner.py \
    --lang ${LANG:-"en"} \
    --train_corpus_path ${TRAIN_CORPUS_PATH:-"$DATA_ROOT/bc5cdr/cdr_train_corpus.txt"} \
    --val_corpus_path ${VAL_CORPUS_PATH:-"$DATA_ROOT/bc5cdr/cdr_test_corpus.txt"} \
    --max_length ${MAX_LENGTH:-512} \
    --lm_path ${LM_PATH:-"./tmp/cdr_lm-512"} \
    --seed ${SEED:-8} \
    --mlm_probability ${MLM_PROBABILITY:-0.15} \
    --lm_train_batch_size ${LM_TRAIN_BATCH_SIZE:-4} \
    --max_steps ${MAX_STEPS:-1000} \
    --lm_accumulation_steps ${LM_ACCUMULATION_STEPS:-4} )

if [[ -v WANDB ]]; then
    cmd+=( --wandb )
fi

echo ${green}=== Pretraining ===${reset}
"${cmd[@]}"
echo ${green}=== Done ===${reset}
