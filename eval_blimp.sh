#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)
MODEL_REVISION=${2:-main}

python -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,backend="causal",revision=$MODEL_REVISION \
    --tasks blimp_supplement \
    --device cpu \
    --batch_size 1 \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json \
    --log_samples \

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.