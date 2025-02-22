#!/bin/bash

LANG_MODEL=$1
LR=${2:-5e-5}           # default: 5e-5
PATIENCE=${3:-10}       # default: 3
BSZ=${4:-64}            # default: 64
MAX_EPOCHS=${5:-10}     # default: 10
SEED=${6:-13}           # default: 13
TASKS=${7:-"boolq,cola,mrpc,multirc,qnli,qqp,rte,sst2,wsc,mnli,mnli-mm"}
TOKS=${8:-"bnc_spoken_h,bnc_spoken_l,gutenberg_h,gutenberg_l,open_subtitles_h,open_subtitles_l,simple_wiki_h,simple_wiki_l,switchboard_l,mix_h,mix_l,childes_h,childes_l,glue"}

# NOTE: if you've already run finetuning and just want to generate predictions,
# you can set `--model_name_or_path "results/finetune/$model_basename/$TRAIN_NAME/"`
# and remove the `--do_train` and `--do_eval` arguments.

# for task in {boolq,cola,mrpc,multirc,qnli,qqp,rte,sst2,wsc}; do
# for task in {mnli,mnli-mm,}; do

# for tok in bnc_spoken_h bnc_spoken_l gutenberg_h gutenberg_l open_subtitles_h open_subtitles_l simple_wiki_h simple_wiki_l switchboard_l mix_h mix_l childes_h childes_l glue; do

# for tok in bnc_spoken_h bnc_spoken_l gutenberg_h gutenberg_l open_subtitles_h open_subtitles_l simple_wiki_h simple_wiki_l; do # need to do mnli,mnli-mm    
for tok in $(echo $TOKS | tr "," "\n"); 
do
    #model_basename="$(basename $MODEL_PATH)_seed$SEED"
    model_basename="llama${LANG_MODEL}_${tok}_ftseed$SEED"

    # for task in {mnli,mnli-mm}; do
    # for task in {boolq,cola,mrpc,multirc,qnli,qqp,rte,sst2,wsc,mnli,mnli-mm}; do
    for task in $(echo $TASKS | tr "," "\n")
    do
        echo $task

    	if [[ $task = "mnli-mm" ]]; then
    		TRAIN_NAME="mnli"
    		VALID_NAME="mnli-mm"
    		DO_TRAIN=False
    		MODEL_PATH_FULL="results/finetune/$model_basename/$TRAIN_NAME/"
    	else
    		TRAIN_NAME=$task
    		VALID_NAME=$task
    		DO_TRAIN=True
    		MODEL_PATH_FULL="hrasto/llama${LANG_MODEL}_${tok}"
    	fi
    
    	mkdir -p results/finetune/$model_basename/$task/
    
    	python finetune_classification.py \
    	  --model_name_or_path $MODEL_PATH_FULL \
    	  --output_dir results/finetune/$model_basename/$task/ \
    	  --train_file evaluation_data/glue_filtered/$TRAIN_NAME.train.jsonl \
    	  --validation_file evaluation_data/glue_filtered/$VALID_NAME.valid.jsonl \
    	  --do_train $DO_TRAIN \
    	  --do_eval \
    	  --do_predict \
    	  --max_seq_length 128 \
    	  --per_device_train_batch_size $BSZ \
    	  --learning_rate $LR \
    	  --num_train_epochs $MAX_EPOCHS \
    	  --patience $PATIENCE \
    	  --evaluation_strategy epoch \
    	  --save_strategy epoch \
    	  --overwrite_output_dir \
    	  --seed $SEED
       
        if [[ $task = "mnli-mm" ]]; then
            rm -rf results/finetune/$model_basename/*
        fi
    done
done

# Add `--trust_remote_code` if you need to load custom config/model files.
# If you run into memory issues, try reducing $BSZ or reducing `--max_seq_length` first.