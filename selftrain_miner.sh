#!/bin/bash

# LANG="en"
# TRAIN_DATA_PATH=""
# GAZETTEERS_PATH=""
# LABELS_PATH=""
LM_PATH="./tmp/cdr_lm"
# MAX_LENGTH=0
# NER_BATCH_SIZE=0
# LR=0.0
# MOMENTUM=0.0
# PATIENCE=0
# NER_EPOCHS=0
# NER_ACCUMULATION_STEPS=0
# NER_PATH=""
# MIN_DELTA=
# CORRECTED_LOSS=
# DORPOUT=0.2
# GAMMA=
# SEED=0

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Training ===${reset}
python3 selftrain_miner.py \
    --lang ${LANG:-"en"} \
    --train_data_path ${TRAIN_DATA_PATH:-"./data/bc5cdr/distant/cdr_train.conll"} \
    --val_data_path ${VAL_DATA_PATH:-"./data/bc5cdr/gold/cdr_test.conll"} \
    --gazetteers_path ${GAZETTEERS_PATH:-"./data/bc5cdr/gazetteers/"} \
    --labels_path ${LABELS_PATH:-"./data/bc5cdr/labels.txt"} \
    --lm_path ${LM_PATH:-"roberta-base"} \
    --max_length ${MAX_LENGTH:-256} \
    --ner_batch_size ${NER_BATCH_SIZE:-4} \
    --lr ${LR:-0.0001} \
    --ner_epochs ${NER_EPOCHS:-50} \
    --ner_accumulation_steps ${NER_ACCUMULATION_STEPS:-4} \
    --ner_path ${NER_PATH:-"./tmp/ner.pt"} \
    --corrected_loss ${CORRECTED_LOSS:-0} \
    --gamma ${GAMMA:-0.75} \
    --sam ${SAM:-0} \
    --dropout ${DROPOUT:-0.2} \
    --seed ${SEED:-8}
echo ${green}--- Done ---${reset}
