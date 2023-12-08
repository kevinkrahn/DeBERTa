# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import re
import sentencepiece as sp
from ..utils import get_logger
logger = get_logger()


__all__ = ['SPMCharTokenizer']

class SPMCharTokenizer:
  def __init__(self, vocab_file, special_tokens=[]):
    assert os.path.exists(vocab_file)

    self.spm = sp.SentencePieceProcessor()
    self.spm.load(vocab_file)
    vocab_size = self.spm.GetPieceSize()
    self.vocab = {self.spm.IdToPiece(i):i for i in range(vocab_size)}
    self.id_to_tokens = [self.spm.IdToPiece(i) for i in range(vocab_size)]
    self.special_tokens = ['[PAD]', '[CLS]', '[WORD_CLS]', '[SEP]', '[UNK]', '[MASK]', *special_tokens]
    self.special_token_indices = [self.vocab[token] for token in self.special_tokens]

    self.pad_id = self.vocab['[PAD]']
    self.cls_id = self.vocab['[CLS]']
    self.word_cls_id = self.vocab['[WORD_CLS]']
    self.sep_id = self.vocab['[SEP]']
    self.unk_id = self.vocab['[UNK]']
    self.mask_id = self.vocab['[MASK]']

  def tokenize(self, text):
    return self.spm.encode_as_ids(text)

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
