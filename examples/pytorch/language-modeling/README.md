<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->



# ICEBERT Language model training

This repository guides you through the training of an ICEBERT-base model.
The files necessary for the training are already provided in the repo. If you want to create those files from scratch, follow the [icebert clustering notebook](https://github.com/BassaniRiccardo/transformers/blob/master/examples/pytorch/language-modeling/icebert/clustering_notebooks/icebert_clustering.ipynb).

The ICEBERT model builds on the BERT model, but includes 9 different languages:
{Arabic, Bengali, English, Finnish, Indonesian, Korean, Russian, Swahili, Telugu}.

## Requirements

The torch xla distribution is required for TPUs training. Need free TPUs? Check https://www.tensorflow.org/tfrc .

## Setup

### Virtual Cluster Creation

NOTE: Be aware of GCP costs, escpeciall for TPUs. Also, make sure you storage bucket is in the same zone of your machines.

    gcloud config set project YOUR_PROJECT
    gcloud config set compute/zone YOUR_ZONE
    gcloud compute tpus execution-groups create --name=GROUP_NAME --zone=YOUR_ZONE --tf-version=2.4.1 --disk-size=1000GB --machine-type=n2d-highmem-8 --accelerator-type=v3-8


### Setup Python

    sudo apt update 
    sudo apt install python3 python3-dev python3-venv 
    sudo apt-get install wget 
    wget https://bootstrap.pypa.io/get-pip.py 
    sudo python3 get-pip.py
    python3 -m venv prep
    source prep/bin/activate		


### Install required packages

    sudo apt install git
    pip install -U pip
    pip install git+https://github.com/huggingface/transformers
    pip install numpy
    pip install tqdm
    pip install datasets
    pip install sentencepiece
    pip install protobuf
    pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8.1-cp37-cp37m-linux_x86_64.whl
    pip install torch


### Mount your Bucket

    mkdir bucket
    export GCSFUSE_REPO=gcsfuse-bionic main
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install gcsfuse
    gcsfuse --implicit-dirs icebert PATH_WHERE_YOUR_BUCKET_WILL_BE_MOUNTED



## Data preparation

1. Download wikipedia Dumps.
2. Extract .xml dumps to .txt files using the WikiExtractor script. Use only smaller chunks for the small corpus creation.

**Steps 3-6 can be performed by running the desired script from the bash_script folder.**
Change the root of the directory variables according to your directory structure.

3. For each language, concatenate short sentences in the corpus so that all lines contain a number of tokens close to MAX_SEQ.
This must be done manually here, since TPUs require the --line_by_line flag. While doing this, also lowercase the baseline corpus and create the cID corpus for each language. 

4. Compute the number of lines per corpus:

5. Get the oversampled number of lines (default ALPHA=0.3) per language:

6. Create a single large corpus. English lines are mantained, while other languages' ones are duplicated.
Shuffle the obtained corpus. Do this for both the baseline and the model. 

For multiprocessing:
- add the argument --multiprocessing
- specify --mp_input_folder, the absolute path to the folder containing txt data split into smaller files 



## Training

Scripts for small-bert and bert-base training can be found in the **bash_scripts folder**. 

Training arguments are listed at https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py . 

The current scripts yield an undertrained ICEBERT-base model.
The training takes around 5 days with the given script, an n2d-highmem-8 vcpu and a single v3-8 TPU. 
Increment the number of training steps or the batch size for a more performing model.
This would probably require using a TPU-pod.




<br />
<br />
<br />
<br />

<h3 align="center"> Below you find the original README from huggingface </h3>

<br />
<br />
<br />
<br />

# HuggingFace general language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2,
ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling
(CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM)
loss. XLNet uses permutation language modeling (PLM), you can find more information about the differences between those
objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

**Note:** The old script `run_language_modeling.py` is still available [here](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py).

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_clm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```bash
python run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-clm
```

### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore,
converge slightly slower (over-fitting takes more epochs).

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

To run on your own training and validation files, use the following command:

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_mlm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```bash
python run_mlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path roberta-base \
    --output_dir /tmp/test-mlm
```

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.

### Whole word masking

This part was moved to `examples/research_projects/mlm_wwm`.

### XLNet and permutation language modeling

XLNet uses a different training objective, which is permutation language modeling. It is an autoregressive method
to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input
sequence factorization order.

We use the `--plm_probability` flag to define the ratio of length of a span of masked tokens to surrounding
context length for permutation language modeling.

The `--max_span_length` flag may also be used to limit the length of a span of masked tokens used
for permutation language modeling.

Here is how to fine-tune XLNet on wikitext-2:

```bash
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

To fine-tune it on your own training and validation file, run:

```bash
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.
