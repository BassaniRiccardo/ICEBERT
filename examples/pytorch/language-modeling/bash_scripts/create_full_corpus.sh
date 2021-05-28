#!/bin/bash

# Create a training corpus.
# IMPORTANT: DATA_FOLDER/original must contain a ln.txt file for each language (e.g. ar.txt)

MAX_SEQ=512
ICEBERT_FOLDER="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/icebert"
DATA_FOLDER="/content/drive/MyDrive/Thesis/transformers/examples/pytorch/language-modeling/data"

# create dense corpora, both for baseline and model(cid-mapped) 
python language-modeling/create_oversampled_wikicorpus.py preprocess --max_seq $MAX_SEQ \
    --icebert_folder $ICEBERT_FOLDER \
    --data_folder $DATA_FOLDER \

# save a dictionary with the number of lines in each language's dense corpus (counting baseline or model lines is the same)
python create_oversampled_wikicorpus.py  count_lines --max_seq $MAX_SEQ \
    --icebert_folder $ICEBERT_FOLDER\
    --data_folder $DATA_FOLDER 

# derive the number of lines for each language to mimic exponential oversampling, and save them into a dictionary
# also save the oversampling factors for each language, ropunded to integers, into a dictionary
python create_oversampled_wikicorpus.py  derive_lines --ALPHA=0.3 --max_seq $MAX_SEQ \
    --icebert_folder $ICEBERT_FOLDER\
    --data_folder $DATA_FOLDER 

# create oversampled file for the baseline
python create_oversampled_wikicorpus.py  oversample --max_seq $MAX_SEQ \
--icebert_folder $ICEBERT_FOLDER\
--data_folder $DATA_FOLDER \

# create oversampled file for the corpus
python create_oversampled_wikicorpus.py  oversample --max_seq $MAX_SEQ \
--icebert_folder $ICEBERT_FOLDER\
--data_folder $DATA_FOLDER \
--cid