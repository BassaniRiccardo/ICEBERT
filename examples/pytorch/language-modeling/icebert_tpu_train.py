import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import logging
import traceback
import math
import os
import sys
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.7.0.dev0")

# Set up logger: writing both to file and to std output
file_handler = logging.FileHandler(filename='tpu_training_logger')
file_handler.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        handlers=handlers
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# avoid creating useless and space-consuming copies of the data for each tpu-core
SERIAL_EXEC = xmp.MpSerialExecutor()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_type: Optional[str] = field(
        default="cased_baseline",
        metadata={"help" : "uncased_baseline,  cased_baseline, or model"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def add_custom_args(hf_parser):
    hf_parser.add_argument(
            '--icebert_folder',
            type=str,
            default="/home/riccardobassani17/bucket/transformers/examples/pytorch/language-modeling/icebert",
            help="Path to folder containing icebert utils and files"
    )  
    hf_parser.add_argument(
            '--config_file',
            type=str,
            default="/home/riccardobassani17/bucket/transformers/examples/pytorch/language-modeling/icebert/config_files/small_bert.json",
            help="Path of the BertConfig json file, relative to the icebert folder"
        )

    return hf_parser

def get_tokenized_dataset():
    tokenized_datasets = datasets.load_dataset('text', data_files=data_files, cache_dir=cache_dir)
    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_special_tokens_mask=True,
        )
    return tokenized_datasets.with_transform(tokenize_function)

def get_data_collator():
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

def map_fn(index):  

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = add_custom_args( HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)) )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()
            
    logger.info(f"parser built")
    # load and instantiate tokenizer
    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained( (Path(args.icebert_folder) / (str(data_args.max_seq_length) + "_tokenizers") / (model_args.model_type + "_tokenizer")))

    # load and instantiate configuration file
    with open(args.config_file, 'r') as fp:
        config_dict = json.load(fp)
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = BertConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=data_args.max_seq_length, \
                        hidden_size=config_dict["hidden_size"], num_hidden_layers=config_dict["num_hidden_layers"], \
                        num_attention_heads=config_dict["num_attention_heads"], intermediate_size=config_dict["intermediate_size"], \
                        hidden_act=config_dict["hidden_act"], hidden_dropout_prob=config_dict["hidden_dropout_prob"], \
                        attention_probs_dropout_prob=config_dict["attention_probs_dropout_prob"], type_vocab_size=config_dict["type_vocab_size"], \
                        initializer_range=config_dict["initializer_range"], layer_norm_eps=config_dict["layer_norm_eps"], **config_kwargs)

    # load and instantiate model
    # IMPORTANT: the model is wrapped using the xmp.MpModelWrapper, which loads the model only once, in the global scope
    model = xmp.MpModelWrapper(BertForMaskedLM(config))

    logger.info(f"tokenizer and model instantiated")

    # move model to device
    device = xm.xla_device()
    model.to(device)
    xm.rendezvous("Model moved to device")

    # prepare dataset and datacollator for on-the-fly tokenization and masking
    global data_files
    data_files = {"train": data_args.train_file}
    global max_len
    max_len = data_args.max_seq_length
    global cache_dir
    cache_dir = model_args.cache_dir
    tokenized_datasets = SERIAL_EXEC.run(get_tokenized_dataset)
    xm.rendezvous("Tokenized dataset loaded")
    data_collator = SERIAL_EXEC.run(get_data_collator)
    xm.rendezvous("DataCollator loaded")

    # handle possible checkpoints
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # select and optionally sample the train_dataset
    if training_args.do_train:
      if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
      train_dataset = tokenized_datasets["train"]
      if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # setup training parameters
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # start training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info("*** Starting training ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        logger.info("*** Model saved ***")
        try:
            metrics = train_result.metrics
            logger.info("*** metrics assigned from train_result ***")
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            logger.info("*** max train samples assigned ***")
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            logger.info("*** metrics[train_samples] assigned ***")
            trainer.log_metrics("train", metrics)
            logger.info("*** trainer.log_metrics called ***")
            trainer.save_metrics("train", metrics)
            logger.info("*** trainer.save_metrics called ***")
            trainer.save_state()
            logger.info("*** trainer.save_state called: last line in the map_fn function! ***")
        except:
            logger.warning("*** Failed to save metrics and trainer state: check the following exception: ***")
            traceback.print_exc()

if __name__ == "__main__":
    xmp.spawn(map_fn, args=(), nprocs=8, start_method='fork')
