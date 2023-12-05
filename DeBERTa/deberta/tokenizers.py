#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#

""" tokenizers
"""

from .spm_tokenizer import *
from .gpt2_tokenizer import GPT2Tokenizer
from .char_tokenizer import CharTokenizer

__all__ = ['tokenizers']
tokenizers={
    'gpt2': GPT2Tokenizer,
    'spm': SPMTokenizer,
    'char': CharTokenizer,
    }
