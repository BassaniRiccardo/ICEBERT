import sys
import os
import logging
import numpy as np
from tqdm import tqdm
import nltk
import pickle
import json
import random
from pathlib import Path
from argparse import REMAINDER, ArgumentParser

from icebert.cid_mapping import fast_tokenize, encode_cID
from transformers import BertTokenizerFast

# TODO add the actions, follow the README

logging.basicConfig(filename="logger",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
logger = logging.getLogger(__name__)

class NLTKSegmenter:
    def __init__(self):
        pass

    def segment_string(self, line, lowercase=True):
        sentences = nltk.tokenize.sent_tokenize(line)
        if lowercase:
            lc_sentences = [s.lower() for s in sentences]
            return lc_sentences
        return sentences


def parse_args():
    parser = ArgumentParser(
        description=(
            "PyTorch TPU distributed training launch "
            "helper utility that will spawn up "
            "multiple distributed processes"
        )
    )

    # Main argument
    parser.add_argument("action", type=str, help="The action to perform. Choose between: \"preprocess\" \"count_lines\" \"derive_lines\" \"oversample\"")
    
    # Data related arguments
    parser.add_argument(
        '--data_folder',
        type=str,
        help='Specify absolute path to the folder containg data.',
        default="transformers/examples/pytorch/language-modeling/data"
    )
    parser.add_argument(
        '--languages',
        type=str,
        help='Specify the languages as 2-letters code separated by commas, without spaces.',
        default="ar,bn,en,fi,id,ko,ru,sw,te"
    )
    parser.add_argument("--max_seq", type=int, help="The maximum number of tokens in a sequence", default=512)
    parser.add_argument("--ALPHA", type=float, help="The alpha parameter of the exponential sampling. The closer to zero, the higher the oversampling", default=0.3)

    # Arguments for the cID mapping
    parser.add_argument('--cid', action='store_true')
    parser.add_argument(
        '--monolingual_tokenizers_root_path',
        type=str,
        help='Specify name of the monolingual tokenizers vocabularies. The language specific files are supposed to be named \"ln.txt\".',
        default="monolingual_tokenizers"
    )
    parser.add_argument(
        '--cid_mapper_pickle_path',
        type=str,
        help='Specify name of the pickle file containing the cID mapper.',
        default="tok_to_cID_string.pkl"
    )
    parser.add_argument(
        '--icebert_folder',
        type=str,
        help='Specify absolute path to the folder containg icebert utils and files.',
        default="/Users/ricca/Thesis/transformers/examples/pytorch/language-modeling/icebert"
    )

    return parser.parse_args()


def get_corpus(ln, dense=True, cid=False):
    if cid:
        return Path(args.data_folder) / ("cID_dense_wiki_" + ln + "_" + str(args.max_seq) + ".txt")
    if dense:
        return Path(args.data_folder) / ("dense_wiki_" + ln + "_" + str(args.max_seq) + ".txt")
    return Path(args.data_folder) / (ln + ".txt")


def preprocess_corpus(ln, mapper, max_seq):
    """
    Merge/separate lines so that every line is made of approximately max_seq tokens. Create:
        - a new baseline corpus, lowercased.
        - a new cID corpus, made of cID-strings.
    """
    original_corpus_path=get_corpus(ln, dense=False, cid=False)
    dense_corpus_path=get_corpus(ln, dense=True, cid=False)
    cID_dense_corpus_path=get_corpus(ln, dense=False, cid=True)
    

    with open(original_corpus_path, "r", encoding='utf-8') as original_corpus, \
            open(dense_corpus_path, "x", encoding='utf-8') as dense_corpus, \
            open(cID_dense_corpus_path, "x", encoding='utf-8') as cID_dense_corpus:
            lines_list = original_corpus.read().splitlines()
            tokenizer = BertTokenizerFast(Path(args.icebert_folder) / args.monolingual_tokenizers_root_path / (ln + '.txt'), do_lower_case=False, add_special_tokens = True)
            sentence_tokenizer = NLTKSegmenter()
            dense_line = []
            cID_dense_line = []
            line_length = 0
            for line in tqdm(lines_list):
                sentences = sentence_tokenizer.segment_string(line)
                # we work at sentence level (not line level, to avoid cutting very long lines)
                for sentence in sentences:
                    cIDs = encode_cID(fast_tokenize(sentence, ln, tokenizer, mark=True), mapper)
                    line_length += len(cIDs)
                    dense_line.append(sentence.strip())
                    cID_dense_line.append(" ".join(cIDs).strip())
                    # if we reach the maximum number of tokens, we wrote down the dense sentence and start building a new one.
                    if line_length > max_seq:
                        dense_corpus.write(" ".join(dense_line) + "\n")
                        cID_dense_corpus.write(" ".join(cID_dense_line) + "\n")
                        dense_line = []
                        cID_dense_line = []
                        line_length = 0
       
    return


def count_lines(ln, dense=True, cid=False, get_longest_line=False):
    with open(get_corpus(ln, dense, cid), "r", encoding='utf-8') as f:
        lines = f.readlines()
        if get_longest_line:
            max_len = 0
            for l in lines:
                line_len = len(l.split())
                if line_len > max_len:
                    max_len = line_len
            print("The longest line has {} words".format(max_len))
        return len(lines)


def exp_ratio(lines_voc, alpha, langs):
  original_ratios = {}
  corrected_ratios = {}
  oversampled_lines_voc = {}
  sizes = np.array(list(lines_voc.values()))
  ratios = sizes / np.sum(sizes)
  exp_ratios = ratios**alpha
  normalized_exp_ratios = exp_ratios / np.sum(exp_ratios)
  # convert the lists to dictionaries
  for i, ln in enumerate(langs):
    original_ratios[ln] = ratios[i]
    corrected_ratios[ln] = normalized_exp_ratios[i]
  for i, ln in enumerate(langs):
    oversampled_lines_voc[ln] = int(lines_voc["en"] * (corrected_ratios[ln] / corrected_ratios["en"]) )
  return original_ratios, corrected_ratios, oversampled_lines_voc


def create_oversampled_file(oversampling_factors):
    """
    Creates a single file containing shuffled lines in all languages, so that the number of lines per language
    respects the exponential sampling ratios.
    """
    union_lines_list = []
    for ln in args.languages.split(','):
        logger.info(f"oversampling {ln}...")
        corpus_path = get_corpus(ln, cid=args.cid)
        with open(corpus_path, "r") as corpus:
            all_lines = corpus.readlines()
            union_lines_list.extend(all_lines * oversampling_factors[ln])
    # shuffle it to avoid bias toward first languages (NOTE: no problem since we are not doing NSP)
    # we want to keep the same shuffled indexes to minimize the confounds between baseline and model
    logger.info(f"generating shuffling indexes...")
    indexes = load_or_generate_shuffle_indexes(len(union_lines_list))
    logger.info(f"shuffling lines list...")
    shuffled_line_list = [union_lines_list[index] for index in indexes]
    outfile = "model_training_corpus.txt" if args.cid else "baseline_training_corpus.txt"
    logger.info(f"writing lines to file...")
    with open(Path(args.data_folder) / outfile, "x") as out:
        for line in tqdm(shuffled_line_list):
            out.write(line)


def load_or_generate_shuffle_indexes(length):
    path = Path(args.data_folder) / "shuffled_indexes.pkl"
    if os.path.isfile(path):
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception: # so many things could go wrong, can't be more specific.
                pass 
    with open(path, "wb") as f:
        indexes = list(range(length))
        random.shuffle(indexes)
        pickle.dump(indexes, f)
    return indexes


def main(args):
    
    if args.action == "preprocess":
        nltk.download('punkt')
        cid_mapper = pickle.load(open(Path(args.icebert_folder) / args.cid_mapper_pickle_path, "rb"))
        for ln in args.languages.split(","):
            preprocess_corpus(ln, cid_mapper, args.max_seq)
    elif args.action == "count_lines":
        lines_voc = {}
        for ln in args.languages.split(","):
            lines_voc[ln] = count_lines(ln, dense=True, cid=False)
        with open(Path(args.data_folder) / ('original_lines_numbers_' + str(args.max_seq) + '.json'), 'w') as fp:
            json.dump(lines_voc, fp)
    elif args.action == "derive_lines":
        with open(Path(args.data_folder) /  ('original_lines_numbers_' + str(args.max_seq) + '.json'), 'r') as fp:
            lines_voc = json.load(fp)
        original_ratios, corrected_ratios, oversampled_lines_voc = exp_ratio(lines_voc, args.ALPHA, args.languages.split(","))
        logger.info(f"original_ratios: {original_ratios}")
        logger.info(f"corrected_ratios: {corrected_ratios}")
        with open(Path(args.data_folder) / ('oversampled_lines_numbers_' + str(args.max_seq) + '.json'), 'w') as fp:
            json.dump(oversampled_lines_voc, fp)
        oversampling_factors = {}
        for ln in args.languages.split(","):
            oversampling_factors[ln] = int(round(oversampled_lines_voc[ln] / lines_voc[ln]))
        with open(Path(args.data_folder) / ('oversampling_factors_' + str(args.max_seq) + '.json'), 'w') as fp:
            json.dump(oversampling_factors, fp)
    elif args.action == "oversample":
        with open(Path(args.data_folder) / ('oversampling_factors_' + str(args.max_seq) + '.json'), 'r') as fp:
            oversampling_factors = json.load(fp)
        create_oversampled_file(oversampling_factors)
    # here debugging actions
    elif args.actions == "preprocess_test_corpus":
        nltk.download('punkt')
        cid_mapper = pickle.load(open(Path(args.icebert_folder) / args.cid_mapper_pickle_path, "rb"))
        create_dense_corpora(original_corpus_path, dense_corpus_path, cID_dense_corpus_path, mapper, ln, max_seq)
    elif args.action == "longest_line_sw":
        count_lines("sw", get_longest_line=True)
    else:
        print("Insert a valid action")


if __name__ == "__main__":
    args = parse_args()
    main(args)