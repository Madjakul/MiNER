#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
TRAIN_DATA_PATH="./data/ncbi_disease/distant/ncbi_train.conll"
LABELS_PATH="./data/ncbi_disease/labels.txt"
LM_PATH="./tmp/ncbi_lm-256"
MAX_LENGTH=256
NER_BATCH_SIZE=8
# LR=0.0005
# MOMENTUM=0.0
# CLIP=
# PATIENCE=5
# NER_EPOCHS=15
NER_ACCUMULATION_STEPS=2
NER_PATH="./tmp/ncbi_ner-256.pt"
# DORPOUT=0.2
# SEED=0

VAL_DATA_PATH="./data/ncbi_disease/gold/ncbi_dev.conll"
SAM=1
# Q=0.6
LOSS_FN="c_nll"
WANDB=

# *****************************************************************************


green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

cmd=( python3 train_miner.py \
    --lang ${LANG:-"en"} \
    --train_data_path ${TRAIN_DATA_PATH:-"./data/bc5cdr/distant/cdr_train.conll"} \
    --labels_path ${LABELS_PATH:-"./data/bc5cdr/labels.txt"} \
    --lm_path ${LM_PATH:-"./tmp/cdr_lm-512"} \
    --max_length ${MAX_LENGTH:-512} \
    --ner_batch_size ${NER_BATCH_SIZE:-4} \
    --lr ${LR:-0.005} \
    --momentum ${MOMENTUM:-0.9} \
    --clip ${CLIP:-5.0} \
    --patience ${PATIENCE:-10} \
    --ner_epochs ${NER_EPOCHS:-15} \
    --ner_accumulation_steps ${NER_ACCUMULATION_STEPS:-2} \
    --ner_path ${NER_PATH:-"./tmp/cdr_ner-512.pt"} \
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
if [[ -v LOSS_FN ]]; then
    cmd+=( --loss_fn $LOSS_FN )
fi
if [[ -v Q ]]; then
    cmd+=( --q $Q )
fi

echo ${green}=== Training Miner ===${reset}
"${cmd[@]}"
echo ${green}=== Done ===${reset}

