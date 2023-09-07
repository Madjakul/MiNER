#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="en"
# TRAIN_DATA_PATH="$DATA_ROOT/wikigold/distant/wiki_train.conll"
# LABELS_PATH=""
# LM_PATH="$PROJECT_ROOT/tmp/wiki_lm-128"
# MAX_LENGTH=128
# NER_BATCH_SIZE=4
# ACCUMULATION_STEPS=8
# LR=0.0005
# NER_EPOCHS=10
# NER_PATH="$PROJECT_ROOT/tmp/wiki_ner-128.pt"
# SMOOTH_NER_PATH=""
# DORPOUT=0.2
# SEED=1

# VAL_DATA_PATH="$DATA_ROOT/wikigold/gold/wiki_dev.conll"
# WANDB=
# PROJECT="miner"
# ENTITY="madjakul"

# *****************************************************************************


green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

cmd=( python3 train_smooth_ner.py \
    --lang ${LANG:-"en"} \
    --train_data_path ${TRAIN_DATA_PATH:-"$DATA_ROOT/bc5cdr/distant/cdr_train.conll"} \
    --labels_path ${LABELS_PATH:-"$DATA_ROOT/bc5cdr/distant/labels.txt"} \
    --lm_path ${LM_PATH:-"$PROJECT_ROOT/tmp/cdr_lm-512"} \
    --max_length ${MAX_LENGTH:-512} \
    --ner_batch_size ${NER_BATCH_SIZE:-4} \
    --accumulation_steps ${ACCUMULATION_STEPS:-8} \
    --lr ${LR:-0.0000005} \
    --ner_epochs ${NER_EPOCHS:-5} \
    --partial_ner_path ${NER_PATH:-"$PROJECT_ROOT/tmp/cdr_ner-512.pt"} \
    --smooth_ner_path ${SMOOTH_NER_PATH:-"$PROJECT_ROOT/tmp/cdr_smooth-ner-512"} \
    --dropout ${DROPOUT:-0.1} \
    --seed ${SEED:-8} )

if [[ -v WANDB ]]; then
    cmd+=( --wandb \
        --project $PROJECT \
        --entity $ENTITY )
fi
if [[ -v VAL_DATA_PATH ]]; then
    cmd+=( --val_data_path $VAL_DATA_PATH )
fi

echo ${green}=== Training Miner ===${reset}
"${cmd[@]}"
echo ${green}=== Done ===${reset}

