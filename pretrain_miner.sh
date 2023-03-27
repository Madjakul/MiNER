#!/bin/bash

# LANG="en"
# TRAIN_CORPUS_PATH=""
# VAL_CORPUS_PATH=""
# MAX_LENGTH=0
# LM_PATH=""
# SEED=0
# MLM_PROBABILITY=0.0
# LM_TRAIN_BATCH_SIZE=0
# LM_EPOCHS=0
# LM_ACCUMULATION_STEPS=0

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Pretraining ===${reset}
python3 pretrain_miner.py \
    --lang ${LANG:-"en"} \
    --train_corpus_path ${TRAIN_CORPUS_PATH:-"./data/bc5cdr/cdr_train.conll"} \
    --val_corpus_path ${VAL_CORPUS_PATH:-"./data/bc5cdr/cdr_val.conll"} \
    --max_length ${MAX_LENGTH:-256} \
    --lm_path ${LM_PATH:-"./tmp/lm"} \
    --seed ${SEED:-8} \
    --mlm_probability ${MLM_PROBABILITY:-0.15} \
    --lm_train_batch_size ${LM_TRAIN_BATCH_SIZE:-4} \
    --lm_epochs ${LM_EPOCHS:-50} \
    --lm_accumulation_steps ${LM_ACCUMULATION_STEPS:-16}
echo ${green}--- Done ---${reset}
