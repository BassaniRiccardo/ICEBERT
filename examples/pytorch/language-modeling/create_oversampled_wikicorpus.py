import numpy as np
from pathlib import Path
from argparse import REMAINDER, ArgumentParser

# TODO add the actions, follow the README

langs = ["ar", "bn", "en", "fi", "id", "ko", "ru", "sw", "te"]
sizes_voc = {"ar" : 1636, "bn" : 522, "en" : 32000, "fi": 745, "id": 600, "ko": 703, "ru" : 6161, "sw": 35, "te" : 498}
lines_voc = {'ar': 6910013, 'bn': 1077801, 'en': 70000000, 'fi': 3767147, 'id': 3560410, 'ko': 3767071, 'ru': 18807018, 'sw': 325114, 'te': 1397295}
ALPHA = 0.3 

def parse_args():
    parser = ArgumentParser(
        description=(
            "PyTorch TPU distributed training launch "
            "helper utility that will spawn up "
            "multiple distributed processes"
        )
    )

    # Main argument
    parser.add_argument("--action", type=str, default="all" , help="The action to perform TODO LIST")

    return parser.parse_args()

def concatenate_short_lines(ln):
    # TODO IMPLEMENT
    return

def count_lines(ln):
    wiki_text = Path("/Users/ricca/Desktop/THESIS/txt/") / ln / ("wiki_" + ln + ".txt")
    print(wiki_text)
    with open(wiki_text, "r", encoding='utf-8') as f:
        print("im in")
        lines = f.readlines()
        return len(lines)

def exp_ratio(sizes, alpha, langs):
  original_ratios = {}
  corrected_ratios = {}
  oversampled_lines_voc = {}
  sizes = np.array(list(sizes.values()))
  # print(sizes)
  # print(len(sizes))
  ratios = sizes / np.sum(sizes)
  # print(normalized_sizes)
  exp_ratios = ratios**alpha
  # print(exp_sizes)
  normalized_exp_ratios = exp_ratios / np.sum(exp_ratios)

  for i, ln in enumerate(langs):
    original_ratios[ln] = ratios[i]
    corrected_ratios[ln] = normalized_exp_ratios[i]
  for i, ln in enumerate(langs):
    oversampled_lines_voc[ln] = int( lines_voc["en"] * (corrected_ratios[ln] / corrected_ratios["en"]) )

  return original_ratios, corrected_ratios, oversampled_lines_voc


def main(args):
    args = parse_args()
    if args.action == "get oversampled numbers of lines":
        original_ratios, corrected_ratios, oversampled_lines_voc = exp_ratio(lines_voc, ALPHA, langs)
        print("original_ratios:", original_ratios)
        print("corrected_ratios:", corrected_ratios)
        print("original number of lines:", lines_voc)
        print("oversampled number of lines:", oversampled_lines_voc)
    else:
        print("insert a valid action")


if __name__ == "__main__":
    main()