#!/bin/bash

# LANG="en"
# TRAIN_DATA_PATH=""
# GAZETTEERS_PATH=""
# LABELS_PATH=""
# LM_PATH=""
# MAX_LENGTH=0
# NER_BATCH_SIZE=0
# LR=0.0
# MOMENTUM=0.0
# PATIENCE=0
# NER_EPOCHS=0
# NER_ACCUMULATION_STEPS=0
# NER_PATH=""
# MIN_DELTA=0
CORRECTED_LOSS=1
GAMMA=0.5
# SEED=0

# -----------------------------------------------------------------------------

green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

echo ${green}=== Training ===${reset}
python3 train_miner.py \
    --lang ${LANG:-"en"} \
    --train_data_path ${TRAIN_DATA_PATH:-"./data/bc5cdr/cdr_train.conll"} \
    --gazetteers_path ${GAZETTEERS_PATH:-"./data/gazetteers/"} \
    --labels_path ${LABELS_PATH:-"./data/bc5cdr/labels.txt"} \
    --lm_path ${LM_PATH:-"roberta-base"} \
    --max_length ${MAX_LENGTH:-256} \
    --ner_batch_size ${NER_BATCH_SIZE:-4} \
    --lr ${LR:-0.005} \
    --momentum ${MOMENTUM:-0.9} \
    --patience ${PATIENCE:-5} \
    --ner_epochs ${NER_EPOCHS:-50} \
    --ner_accumulation_steps ${NER_ACCUMULATION_STEPS:-4} \
    --ner_path ${NER_PATH:-"./tmp/ner.pt"} \
    --min_delta ${MIN_DELTA:-0.1} \
    --corrected_loss ${CORRECTED_LOSS:-0} \
    --gamma ${GAMMA:-1.0} \
    --seed ${SEED:-8}
echo ${green}--- Done ---${reset}
