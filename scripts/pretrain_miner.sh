#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
TRAIN_CORPUS_PATH="$DATA_ROOT/wikigold/wiki_train_corpus.txt"
VAL_CORPUS_PATH="$DATA_ROOT/wikigold/wiki_test_corpus.txt"
MAX_LENGTH=512
LM_PATH="./tmp/wiki_lm-512"
# SEED=0
# MLM_PROBABILITY=0.0
LM_TRAIN_BATCH_SIZE=2
# LM_EPOCHS=0
LM_ACCUMULATION_STEPS=8

WANDB=1

# *****************************************************************************


green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

cmd=( python3 pretrain_miner.py \
    --lang ${LANG:-"en"} \
    --train_corpus_path ${TRAIN_CORPUS_PATH:-"$DATA_ROOT/bc5cdr/cdr_train_corpus.txt"} \
    --val_corpus_path ${VAL_CORPUS_PATH:-"$DATA_ROOT/bc5cdr/cdr_test_corpus.txt"} \
    --max_length ${MAX_LENGTH:-256} \
    --lm_path ${LM_PATH:-"./tmp/cdr_lm-256"} \
    --seed ${SEED:-8} \
    --mlm_probability ${MLM_PROBABILITY:-0.15} \
    --lm_train_batch_size ${LM_TRAIN_BATCH_SIZE:-4} \
    --lm_epochs ${LM_EPOCHS:-100} \
    --lm_accumulation_steps ${LM_ACCUMULATION_STEPS:-8} )

if [[ -v WANDB ]]; then
    cmd+=( --wandb )
fi

echo ${green}=== Pretraining ===${reset}
"${cmd[@]}"
echo ${green}=== Done ===${reset}
