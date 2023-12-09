# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

This repository is a fork of the official implementation of [ **DeBERTa**: **D**ecoding-**e**nhanced **BERT** with Disentangled **A**ttention ](https://arxiv.org/abs/2006.03654) and [DeBERTa V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)

The original DeBERTa implementation has been modified to add support for training language models with character tokenizers. 

# Work in Progress
Additionally, this repository implements the architecture described in [Hierarchical Pre-trained Language Model for Open-vocabulary Language Understanding](https://aclanthology.org/2023.acl-long.200.pdf), combining its advantages with the DeBERTA architecture. The implementation can be found in the files prefixed with `ctw_` and the classes prefixed `CharToWord`. Pretraining can be performed using the `"CharToWord_MLM"` and `"CharToWord_RTD"` tasks.