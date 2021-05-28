#!/bin/bash

# Create a training corpus.
# IMPORTANT: DATA_FOLDER/original must contain a ln.txt file for each language (e.g. ar.txt).
#                   The more lines in these files, the less redundant the created corpus will be.
#                   Note that redundacy is not necessarily bad when the goal is to simulate a larger-scale scenario. 
#            DATA_FOLDER/lines_limits.json ca be modified for different ratios.
#                   The current ratio simulates the one obtained with exponential sampling and alpha=0.3. No further oversampling is needed!
#            DATA_FOLDER/oversampling_factors should be kepts as it is (all ones) to skip oversampling.

MAX_SEQ=512
ICEBERT_FOLDER="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/icebert"
DATA_FOLDER="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/180k_data"

# create dense corpora, both for baseline and model(cid-mapped) 
python language-modeling/create_oversampled_wikicorpus.py preprocess --max_seq $MAX_SEQ \
    --icebert_folder $ICEBERT_FOLDER \
    --data_folder $DATA_FOLDER \
    --lines_limits_path $DATA_FOLDER/lines_limits.json \

# create oversampled file for the baseline
python create_oversampled_wikicorpus.py  oversample --max_seq $MAX_SEQ \
--icebert_folder $ICEBERT_FOLDER\
--data_folder $DATA_FOLDER \

# create oversampled file for the corpus
python create_oversampled_wikicorpus.py  oversample --max_seq $MAX_SEQ \
--icebert_folder $ICEBERT_FOLDER\
--data_folder $DATA_FOLDER \
--cid