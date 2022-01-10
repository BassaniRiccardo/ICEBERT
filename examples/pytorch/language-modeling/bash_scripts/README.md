This folder contains scripts for the training of an ICEBERT model/baseline, both for the small and the base architecture. To run:

- Substitute TPU_IP_ADDRESS=10.93.176.98 with your TPU address.
- Substitute STORAGE_BUCKET=/home/riccardobassani17/bucket with the path where your bucket is mounted

To train a model from checkpoints:

1) Add the environmental variable CHECKPOINT_DIR:

		export CHECKPOINT_DIR=$STORAGE_BUCKET/transformers/examples/pytorch/language-modeling/output_models/my_interrupted_training_folder/my_checkpoint_folder

2) Add the argument:    --resume_from_checkpoint $CHECKPOINT_DIR
