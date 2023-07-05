#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify


# ************************* Customizable Arguments ****************************

# LANG="fr"
# LABEL_COMPLETION=1
CORPUS_PATH="$DATA_ROOT/ncbi_disease/ncbi_train_corpus.txt"
CONLL_PATH="$DATA_ROOT/ncbi_disease/distant/ncbi_train.conll"
GAZETTEERS_PATH="$DATA_ROOT/ncbi_disease/gazetteers/"
# UNK_GAZETTEERS_PATH="$DATA_ROOT/ncbi_disease/gazetteers/UNK.txt"

# *****************************************************************************


green=`tput setaf 2`
reset=`tput sgr0`

mkdir tmp logs

cmd=( python3 preprocess_miner.py \
        --lang ${LANG:-"en"} \
        --corpus_path ${CORPUS_PATH:-"$DATA_ROOT/bc5cdr/cdr_train_corpus.txt"} \
        --conll_path ${CONLL_PATH:-"$DATA_ROOT/bc5cdr/distant/cdr_train.conll"} \
        --gazetteers_path ${GAZETTEERS_PATH:-"$DATA_ROOT/bc5cdr/gazetteers/"} )

if [[ -v LABEL_COMPLETION ]]; then
    echo ${green}=== ouais ouais ===${reset}
    cmd+=( --label_completion ${LABEL_COMPLETION} \
        --unk_gazetteers_path ${UNK_GAZETTEERS_PATH:-"$DATA_ROOT/bc5cdr/gazetteers/UNK.txt"} )
fi

echo ${green}=== Preprocessing Miner ===${reset}
"${cmd[@]}"
echo ${green}=== Done ===${reset}

