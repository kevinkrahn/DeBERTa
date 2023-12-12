# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

This repository is a fork of the official implementation of [ **DeBERTa**: **D**ecoding-**e**nhanced **BERT** with Disentangled **A**ttention ](https://arxiv.org/abs/2006.03654) and [DeBERTa V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)

This repository is a modification of **DeBERTa**, adding a hierarchical architecture which combines the advantages of both character and word tokenizers, with word-level self attention and unlimited vocabulary as described in [From Characters to Words: Hierarchical Pre-trained Language Model for Open-vocabulary Language Understanding](https://aclanthology.org/2023.acl-long.200.pdf). This implementation combines its advantages with the **DeBERTA** architecture. Pass `--token_format char_to_word` to the data preparation and training scripts.

# Supported token formats

`--token_format char`

`--token_format subword`

`--token_format char_to_word`