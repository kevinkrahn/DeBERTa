# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/15/2020
#

import copy
import torch

from .ops import *
from .bert import *
from .cache_utils import load_model_state

__all__ = ['DeBERTa', 'CharToWord_DeBERTa', 'CharToWord_LMPredictionHead']

class DeBERTa(torch.nn.Module):
  """ DeBERTa encoder
  This module is composed of the input embedding layer with stacked transformer layers with disentangled attention.

  Parameters:
    config:
      A model config class instance with the configuration to build a new model. The schema is similar to `BertConfig`, \
          for more details, please refer :class:`~DeBERTa.deberta.ModelConfig`

    pre_trained:
      The pre-trained DeBERTa model, it can be a physical path of a pre-trained DeBERTa model or a released configurations, \
          i.e. [**base, large, base_mnli, large_mnli**]

  """

  def __init__(self, config=None, pre_trained=None):
    super().__init__()
    state = None
    if pre_trained is not None:
      state, model_config = load_model_state(pre_trained)
      if config is not None and model_config is not None:
        for k in config.__dict__:
          if k not in ['hidden_size',
            'intermediate_size',
            'num_attention_heads',
            'num_hidden_layers',
            'vocab_size',
            'max_position_embeddings']:
            model_config.__dict__[k] = config.__dict__[k]
      config = copy.copy(model_config)
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.config = config
    self.pre_trained = pre_trained
    self.apply_state(state)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True, position_ids=None, return_att=False):
    """
    Args:
      input_ids:
        a torch.LongTensor of shape [batch_size, sequence_length] \
      with the word token indices in the vocabulary

      attention_mask:
        an optional parameter for input mask or attention mask.

        - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices \
      selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max \
      input sequence length in the current batch. It's the mask that we typically use for attention when \
      a batch has varying length sentences.

        - If it's an attention mask then it will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. \
      In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence.

      token_type_ids:
        an optional torch.LongTensor of shape [batch_size, sequence_length] with the token \
      types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to \
      a `sentence B` token (see BERT paper for more details).

      output_all_encoded_layers:
        whether to output results of all encoder layers, default, True

    Returns:

      - The output of the stacked transformer layers if `output_all_encoded_layers=True`, else \
      the last layer of stacked transformer layers

      - Attention matrix of self-attention layers if `return_att=True`


    Example::

      # Batch of wordPiece token ids.
      # Each sample was padded with zero to the maxium length of the batch
      input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
      # Mask of valid input ids
      attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

      # DeBERTa model initialized with pretrained base model
      bert = DeBERTa(pre_trained='base')

      encoder_layers = bert(input_ids, attention_mask=attention_mask)

    """

    if attention_mask is None:
      attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    ebd_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids, attention_mask)
    embedding_output = ebd_output['embeddings']
    encoder_output = self.encoder(embedding_output,
                   attention_mask,
                   output_all_encoded_layers=output_all_encoded_layers, return_att = return_att)
    encoder_output.update(ebd_output)
    return encoder_output

  def apply_state(self, state = None):
    """ Load state from previous loaded model state dictionary.

      Args:
        state (:obj:`dict`, optional): State dictionary as the state returned by torch.module.state_dict(), default: `None`. \
            If it's `None`, then will use the pre-trained state loaded via the constructor to re-initialize \
            the `DeBERTa` model
    """
    if self.pre_trained is None and state is None:
      return
    if state is None:
      state, config = load_model_state(self.pre_trained)
      self.config = config
    
    prefix = ''
    for k in state:
      if 'embeddings.' in k:
        if not k.startswith('embeddings.'):
          prefix = k[:k.index('embeddings.')]
        break

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    self._load_from_state_dict(state, prefix = prefix, local_metadata=None, strict=True, missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)


class CharToWord_DeBERTa(torch.nn.Module):
  """ DeBERTa encoder

  TODO: Description

  Parameters:
    config:
      A model config class instance with the configuration to build a new model.

    pre_trained:
      Path to a pre-trained model.

  """

  def __init__(self, config=None, pre_trained=None):
    super().__init__()
    state = None
    if pre_trained is not None:
      state, model_config = load_model_state(pre_trained)
      config = copy.copy(model_config)
    
    self.char_embeddings = torch.nn.Embedding(config.vocab_size, config.intra_word_encoder.hidden_size, padding_idx=0)
    self.char_embedding_layer_norm = LayerNorm(config.intra_word_encoder.hidden_size, config.intra_word_encoder.layer_norm_eps)
    self.char_embedding_dropout = StableDropout(config.hidden_dropout_prob)

    #self.word_position_biased_input = getattr(config.intra_word_encoder, 'position_biased_input', True)
    #self.word_position_embeddings = torch.nn.Embedding(config.inter_word_encoder.max_position_embeddings, config.inter_word_encoder.hidden_size)

    self.config = config
    self.pre_trained = pre_trained
    # TODO: perhaps it would be better for both encoders to have the same hidden size
    # in order to share relative embeddings
    self.intra_word_encoder = BertEncoder(config.intra_word_encoder)
    self.inter_word_encoder = BertEncoder(config.inter_word_encoder)
    self.apply_state(state)

  def forward(self, input_ids, char_input_mask, word_input_mask=None, output_all_encoded_layers=True, position_ids=None, return_att=False):
    """
    Args:
      input_ids:
        a torch.LongTensor of shape [batch_size, num_words, num_chars]

      char_input_mask:
        a torch.LongTensor of shape [batch_size, num_words, num_chars] with indices in [0,1]

      word_input_mask:
        a torch.LongTensor of shape [batch_size, num_words, num_chars] with indices in [0,1]

        It's a mask to be used if the input sequence length is smaller than the max \
      input sequence length in the current batch. It's the mask that we typically use for attention when \
      a batch has varying length sentences.

      token_type_ids:
        an optional torch.LongTensor of shape [batch_size, sequence_length] with the token \
      types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to \
      a `sentence B` token (see BERT paper for more details).

      output_all_encoded_layers:
        whether to output results of all encoder layers, default, True

    Returns:

      - The output of the stacked transformer layers if `output_all_encoded_layers=True`, else \
      the last layer of stacked transformer layers

      - Attention matrix of self-attention layers if `return_att=True`


    Example:
      TODO: Example

    """

    input_embeds = self.char_embeddings(input_ids)

    # Note: This does the same thing as MaskedLayerNorm, which doesn't work with this tensor shape
    input_embeds = self.char_embedding_layer_norm(input_embeds).to(input_embeds)
    input_embeds = input_embeds * char_input_mask.unsqueeze(-1).to(input_embeds)

    input_embeds = self.char_embedding_dropout(input_embeds)

    batch_size, num_word, num_char, hidden_size = input_embeds.shape

    # reshape to attend to intra-word tokens rather than full sequence
    input_embeds = input_embeds.reshape(batch_size * num_word, num_char, hidden_size)
    intra_word_mask = char_input_mask.reshape(batch_size * num_word, num_char)
    intra_word_output = self.intra_word_encoder(input_embeds, intra_word_mask, output_all_encoded_layers=False, return_att=False)
    initial_embeds = intra_word_output['hidden_states'][-1]

    # extract [WORD_CLS] embeddings, which are always at the beginning of each word
    word_embeds = initial_embeds[:,0,:]

    # reshape and extract contextualized inter-word representation
    word_embeds = word_embeds.reshape(batch_size, num_word, hidden_size)
    if word_input_mask is None:
      word_input_mask = torch.cat([torch.ones(batch_size, num_word), torch.zeros(batch_size, self.config.max_position_embeddings - num_word)], dim=1)
    inter_word_output = self.inter_word_encoder(word_embeds, word_input_mask, output_all_encoded_layers=output_all_encoded_layers, return_att=return_att)
    word_embeds = inter_word_output['hidden_states'][-1]
    word_embeds = word_embeds.reshape(batch_size * num_word, 1, hidden_size)

    output = {
      'word_embeds': word_embeds,
      'initial_embeds': initial_embeds,
      'intra_word_mask': intra_word_mask,
      'char_embeds': input_embeds,
      'input_shape': (batch_size, num_word, num_char, hidden_size),
      **inter_word_output,
    }

    return output

  def apply_state(self, state = None):
    """ Load state from previous loaded model state dictionary.

      Args:
        state (:obj:`dict`, optional): State dictionary as the state returned by torch.module.state_dict(), default: `None`. \
            If it's `None`, then will use the pre-trained state loaded via the constructor to re-initialize \
            the `DeBERTa` model
    """
    if self.pre_trained is None and state is None:
      return
    if state is None:
      state, config = load_model_state(self.pre_trained)
      self.config = config
    
    prefix = ''
    for k in state:
      if 'embeddings.' in k:
        if not k.startswith('embeddings.'):
          prefix = k[:k.index('embeddings.')]
        break

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    self._load_from_state_dict(state, prefix = prefix, local_metadata=None, strict=True, missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)


class CharToWord_LMPredictionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.intra_word_encoder.hidden_size, config.vocab_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        #self.LayerNorm = LayerNorm(self.embedding_size, config.layer_norm_eps, elementwise_affine=True)

        intra_word_encoder_config = copy.copy(config.intra_word_encoder)
        intra_word_encoder_config.num_hidden_layers = 1
        # TODO: should share relative embeddings with the main model, otherwise this is a giant waste of parameters and memory
        self.intra_word_encoder = BertEncoder(intra_word_encoder_config)

        # TODO: dropout?

    def forward(self, deberta_output, label_index=None):
        batch_size, num_word, num_char, hidden_size = deberta_output['input_shape']
        word_embeds = deberta_output['word_embeds']
        initial_embeds = deberta_output['initial_embeds']
        intra_word_mask = deberta_output['intra_word_mask']

        # TODO: Add residual connection between initial word embeddings and contextual word embeddings
        # as mentioned in the paper (In appendix, section A.3)

        # concatenate to restore the character-level token sequence
        char_embeds = torch.cat([word_embeds, initial_embeds[:,1:,:]], dim=1)
        intra_word_output = self.intra_word_encoder(char_embeds, intra_word_mask, output_all_encoded_layers=False, return_att=False)
        hidden_states = intra_word_output['hidden_states'][-1]

        if label_index is not None:
          hidden_states = hidden_states.view(-1, hidden_states.size(-1))
          hidden_states = hidden_states.index_select(0, label_index)

        char_logits = self.dense(hidden_states)
        char_logits = self.transform_act_fn(char_logits)

        if label_index is None:
          char_logits = char_logits.reshape(batch_size, num_word, num_char, -1) + self.bias

        # TODO: LayerNorm?

        return char_logits
