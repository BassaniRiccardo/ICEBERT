#!/bin/bash

# Call the tpu training for a small-bert model, according to Dufter & Schutze setup
# Only parametrized args should be modified.

ICEBERT_FOLDER="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/icebert"
CONFIG_FILE="bert_base.json"

MODEL_TYPE="cased_baseline"
TRAIN_FILE="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/data/base_training_corpus_cased_baseline_512.txt"
OUTPUT_DIR="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/output_models/base_tpu_cased_baseline_512"

# MODEL_TYPE="uncased_baseline"
# TRAIN_FILE="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/data/base_training_corpus_uncased_baseline_512.txt"
# OUTPUT_DIR="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/output_models/base_tpu_uncased_baseline_512"

# MODEL_TYPE="model"
# TRAIN_FILE="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/data/base_training_corpus_model_512.txt"
# OUTPUT_DIR="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/output_models/base_tpu_model_512"

python xla_spawn.py \
        --num_cores=8 \
        language-modeling/run_mlm_icebert.py \
            --model_type $MODEL_TYPE \
            --train_file $TRAIN_FILE \
            --overwrite_output_dir False \
            --config_file $ICEBERT_FOLDER/config_files/$CONFIG_FILE \
            --max_seq_length  512 \
            --max_steps 1000000 \
            --learning_rate 1e-4 \
            --warmup_steps 10000 \
            --save_steps 1000 \
            --logging_steps 1000 \
            --adam_beta1 0.9 \
            --adam_beta2 0.999 \
            --adam_epsilon 1e-6 \
            --weight_decay 0.01 \
            --save_total_limit 2 \
            --per_device_train_batch_size 256 \
            --icebert_folder $ICEBERT_FOLDER \
            --do_train True \
            --do_eval False \
            --output_dir $OUTPUT_DIR \