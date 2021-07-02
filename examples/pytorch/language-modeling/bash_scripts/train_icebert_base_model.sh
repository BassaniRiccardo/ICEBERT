
export STORAGE_BUCKET=/home/riccardobassani17/bucket
export MODEL_TYPE=model
export TRAIN_FILE=$STORAGE_BUCKET/transformers/examples/pytorch/language-modeling/data/final_corpora/model_tc_512.txt
export ICEBERT_FOLDER=$STORAGE_BUCKET/transformers/examples/pytorch/language-modeling/icebert
export CONFIG_FILE=bert_base.json
export OUTPUT_DIR=$STORAGE_BUCKET/transformers/examples/pytorch/language-modeling/output_models/resume_base_tpu_model_512
export TPU_IP_ADDRESS=10.93.148.138	
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

nohup python3 icebert_tpu_train.py --model_type $MODEL_TYPE --train_file $TRAIN_FILE --remove_unused_columns False --overwrite_output_dir False --config_file $ICEBERT_FOLDER/config_files/$CONFIG_FILE --max_seq_length 512 --max_steps 1000000 --save_steps 25000 --learning_rate 1e-4 --warmup_steps 10000 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-6 --weight_decay 0.01 --fp16 False --per_device_train_batch_size 8 --icebert_folder $ICEBERT_FOLDER --do_train True --do_eval False --output_dir $OUTPUT_DIR &
