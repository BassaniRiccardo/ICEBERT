#!/bin/bash

# Run TPU training for a small-bert model, according to Dufter & Schutze setup (changes: 10x lines, 0.1x epochs)
# Only parametrized args should be modified.

ICEBERT_FOLDER="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/icebert"
CONFIG_FILE="small_bert.json"

MODEL_TYPE="cased_baseline"
TRAIN_FILE="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/180k_data/final_corpora/cased_baseline_tc_512.txt"
OUTPUT_DIR="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/output_models/small_tpu_cased_baseline_512"

# MODEL_TYPE="uncased_baseline"
# TRAIN_FILE="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/180k_data/final_corpora/uncased_baseline_tc_512.txt"
# OUTPUT_DIR="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/output_models/small_tpu_uncased_baseline_512"

# MODEL_TYPE="model"
# TRAIN_FILE="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/180k_data/final_corpora/model_tc_512.txt"
# OUTPUT_DIR="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/output_models/small_512_tpu_model"

python xla_spawn.py \
        --num_cores=8 \
        language-modeling/run_mlm_icebert.py \
            --model_type $MODEL_TYPE \
            --train_file $TRAIN_FILE \
            --overwrite_output_dir False \
            --config_file $ICEBERT_FOLDER/config_files/$CONFIG_FILE \
            --max_seq_length  512 \
            --num_train_epochs 10 \
            --learning_rate 2e-3 \
            --warmup_steps 50 \
            --adam_beta1 0.9 \
            --adam_beta2 0.98 \
            --adam_epsilon 1e-6 \
            --weight_decay 0.01 \
            --save_total_limit 2 \
            --per_device_train_batch_size 256 \
            --icebert_folder $ICEBERT_FOLDER \
            --do_train True \
            --do_eval False \
            --output_dir $OUTPUT_DIR \