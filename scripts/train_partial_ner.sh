#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
TRAIN_DATA_PATH="$DATA_ROOT/wikigold/distant/wiki_train.conll"
LABELS_PATH="$DATA_ROOT/wikigold/labels.txt"
LM_PATH="$PROJECT_ROOT/tmp/wiki_lm-128"
MAX_LENGTH=128
NER_BATCH_SIZE=4
# LR=0.0005
# MOMENTUM=0.0
# CLIP=
NER_EPOCHS=10
NER_PATH="$PROJECT_ROOT/tmp/wiki_ner-128.pt"
# DORPOUT=0.2
# SEED=1

VAL_DATA_PATH="$DATA_ROOT/wikigold/gold/wiki_dev.conll"
SAM=1
# Q=0.3
LOSS_FN="c_nll"
# WANDB=
# PROJECT="miner"
# ENTITY="madjakul"

# *****************************************************************************


green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

cmd=( python3 train_partial_ner.py \
    --lang ${LANG:-"en"} \
    --train_data_path ${TRAIN_DATA_PATH:-"$DATA_ROOT/bc5cdr/distant/cdr_train.conll"} \
    --labels_path ${LABELS_PATH:-"$DATA_ROOT/bc5cdr/labels.txt"} \
    --lm_path ${LM_PATH:-"$PROJECT_ROOT/tmp/cdr_lm-512"} \
    --max_length ${MAX_LENGTH:-512} \
    --ner_batch_size ${NER_BATCH_SIZE:-4} \
    --lr ${LR:-0.005} \
    --momentum ${MOMENTUM:-0.9} \
    --clip ${CLIP:-5.0} \
    --ner_epochs ${NER_EPOCHS:-5} \
    --ner_path ${NER_PATH:-"$PROJECT_ROOT/tmp/cdr_ner-512.pt"} \
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

