# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import re
from ..utils import get_logger
logger = get_logger()


__all__ = ['CharTokenizer']

class CharTokenizer:
  def __init__(self, vocab_file, special_tokens=[]):
    assert os.path.exists(vocab_file)

    self.special_tokens = ['[PAD]', '[CLS]', '[WORD_CLS]', '[SEP]', '[UNK]', '[MASK]', *special_tokens]

    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        tokens = [*self.special_tokens, *vocab['tokens']]
        self.vocab = {token:i for i, token in enumerate(tokens)}
        self.id_to_tokens = tokens

    self.special_token_indices = [self.vocab[token] for token in self.special_tokens]
    self.pad_id = self.vocab['[PAD]']
    self.cls_id = self.vocab['[CLS]']
    self.word_cls_id = self.vocab['[WORD_CLS]']
    self.sep_id = self.vocab['[SEP]']
    self.unk_id = self.vocab['[UNK]']
    self.mask_id = self.vocab['[MASK]']

  def tokenize(self, text):
    text = re.sub(r'\s+', ' ', text)
    pieces = []
    special_token = ''
    for c in text:
        if c in '[':
            special_token = c
        elif c == ']':
            special_token += c
            if special_token in self.special_tokens:
                pieces.append(special_token)
            else:
                #pieces.extend(c if c in self.vocab else '[UNK]' for c in special_token)
                pieces.extend(self.vocab.get(c, self.unk_id) for c in special_token)
            special_token = ''
        else:
            if len(special_token) == 0:
                #pieces.append(c if c in self.vocab else '[UNK]')
                pieces.append(self.vocab.get(c, self.unk_id))
    #pieces = [c if c in self.vocab else '[UNK]' for c in text]
    return pieces

  def sym(self, id):
    return self.ids_to_tokens[id]

  def id(self, sym):
    return self.vocab[sym] if sym in self.vocab else 1


def make_vocab(source_file, output_file):
    with open(source_file, 'r') as f:
        text = f.read()
        text = re.sub(r'\s+', '', text)
    chars = set(text)
    with open(output_file, 'w') as f:
        json.dump({'tokens':list(chars)}, f, indent=2, ensure_ascii=False)
