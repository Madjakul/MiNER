#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
TRAIN_DATA_PATH="./data/ncbi_disease/distant/ncbi_train.conll"
LABELS_PATH="./data/ncbi_disease/labels.txt"
LM_PATH="./tmp/ncbi_lm-128"
MAX_LENGTH=128
NER_BATCH_SIZE=32
# LR=0.0005
# MOMENTUM=0.0
# CLIP=
# PATIENCE=5
# NER_EPOCHS=15
# NER_ACCUMULATION_STEPS=4
NER_PATH="./tmp/ncbi_ner-128.pt"
# DORPOUT=0.2
# SEED=0

VAL_DATA_PATH="./data/ncbi_disease/gold/ncbi_test.conll"
SAM=1
CORRECTED_LOSS=1

WANDB=1

# *****************************************************************************


green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

cmd=( python3 train_miner.py \
    --lang ${LANG:-"en"} \
    --train_data_path ${TRAIN_DATA_PATH:-"./data/bc5cdr/distant/cdr_train.conll"} \
    --labels_path ${LABELS_PATH:-"./data/bc5cdr/labels.txt"} \
    --lm_path ${LM_PATH:-"./tmp/cdr_lm-256"} \
    --max_length ${MAX_LENGTH:-256} \
    --ner_batch_size ${NER_BATCH_SIZE:-8} \
    --lr ${LR:-0.005} \
    --momentum ${MOMENTUM:-0.9} \
    --clip ${CLIP:-5.0} \
    --patience ${PATIENCE:-5} \
    --ner_epochs ${NER_EPOCHS:-20} \
    --ner_accumulation_steps ${NER_ACCUMULATION_STEPS:-1} \
    --ner_path ${NER_PATH:-"./tmp/cdr_ner-256.pt"} \
    --dropout ${DROPOUT:-0.1} \
    --seed ${SEED:-8} )

if [[ -v WANDB ]]; then
    cmd+=( --wandb \
        --project $PROJECT \
        --entity $ENTITY )
fi
if [[ -v CORRECTED_LOSS ]]; then
    cmd+=( --corrected_loss)
fi
if [[ -v SAM ]]; then
    cmd+=( --sam )
fi
if [[ -v VAL_DATA_PATH ]]; then
    cmd+=( --val_data_path $VAL_DATA_PATH )
fi

echo ${green}=== Training Miner ===${reset}
"${cmd[@]}"
echo ${green}=== Done ===${reset}

