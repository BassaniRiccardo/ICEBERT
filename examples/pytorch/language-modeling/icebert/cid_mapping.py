def fast_tokenize(text, ln, tokenizer, mark=True):
  """
  Tokenizes a text given a language and a tokenizer. In addition to the tokenization provided by the passed tokenizer:
  - lowercases the text;
  - replaces all digits with 0s;
  - marks tokens with a langaguage marker ("_ln").
  Returns a list of strings.
  """
  text = text.lower()
  text = ''.join("0" if c.isdigit() else c for c in text)
  tokens = tokenizer.tokenize(text)
  if mark:
    for i, t in enumerate(tokens):
      tokens[i] = t + '_' + ln            
  return tokens


def encode_cID(tokens, tok_to_cID, verbose=False, do_join=False):
  """
  Encodes a sentence as a list of cluster IDs (list of strings).
  It takes as input a list of tokens. If a token is not in the vocabulary, it tries to unmark it.
  If it still cannot find it, it encodes it as "UNK".
  Returns a list of strings (representing numbers).
  """
  cIDs = []
  for t in tokens:
    try:
      id = tok_to_cID[t]
    except:
      try:
        id = tok_to_cID[t[:-3]]
      except:
        id = "[UNK]"
    cIDs.append(id)
  if verbose:
    print(tokens)
  if do_join:
      return " ".join(cIDs)
  return cIDs
