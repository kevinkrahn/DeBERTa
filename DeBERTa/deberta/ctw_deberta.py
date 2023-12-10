# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import torch

from .ops import *
from .bert import *
from .cache_utils import load_model_state

__all__ = ['CharToWord_DeBERTa', 'CharToWord_LMPredictionHead', 'CharToWord_LMMaskPredictionHead']

class CharToWord_DeBERTa(torch.nn.Module):
  def __init__(self, config=None, pre_trained=None):
    super().__init__()
    state = None
    if pre_trained is not None:
      state, model_config = load_model_state(pre_trained)
      config = copy.copy(model_config)
    
    self.char_embeddings = torch.nn.Embedding(config.vocab_size, config.intra_word_encoder.hidden_size, padding_idx=0)
    self.char_embedding_layer_norm = LayerNorm(config.intra_word_encoder.hidden_size, config.intra_word_encoder.layer_norm_eps)
    self.char_embedding_dropout = StableDropout(config.intra_word_encoder.hidden_dropout_prob)

    #self.position_biased_input = getattr(config.intra_word_encoder, 'position_biased_input', True)
    #self.word_position_embeddings = torch.nn.Embedding(config.inter_word_encoder.max_position_embeddings, config.inter_word_encoder.hidden_size)

    self.config = config
    self.pre_trained = pre_trained
    # TODO: Try sharing relative embeddings between intra and inter word encoders
    self.intra_word_encoder = BertEncoder(config.intra_word_encoder)
    self.inter_word_encoder = BertEncoder(config.inter_word_encoder)
    self.apply_state(state)

  def forward(self, input_ids, char_input_mask, word_input_mask, char_position_ids=None, word_position_ids=None, output_all_encoded_layers=True, return_att=False):
    input_embeds = self.char_embeddings(input_ids)

    input_embeds = self.char_embedding_layer_norm(input_embeds)

    # TODO: Determine why masking the embeddings causes Infs in the loss
    #mask = char_input_mask.unsqueeze(-1).to(input_embeds)
    #input_embeds = input_embeds * mask

    input_embeds = self.char_embedding_dropout(input_embeds)

    batch_size, num_word, num_char, hidden_size = input_embeds.shape

    # reshape to attend to intra-word tokens rather than full sequence
    input_embeds = input_embeds.reshape(batch_size * num_word, num_char, hidden_size)
    intra_word_mask = char_input_mask.reshape(batch_size * num_word, num_char)
    intra_word_output = self.intra_word_encoder(input_embeds, intra_word_mask, output_all_encoded_layers=False, return_att=False)
    initial_embeds = intra_word_output['hidden_states'][-1]

    # extract [WORD_CLS] embeddings, which are always at the beginning of each word
    initial_word_embeds = initial_embeds[:,0,:]

    # reshape and extract contextualized inter-word representation
    word_embeds = initial_word_embeds.reshape(batch_size, num_word, hidden_size)
    inter_word_output = self.inter_word_encoder(word_embeds, word_input_mask, output_all_encoded_layers=output_all_encoded_layers, return_att=return_att)
    word_embeds = inter_word_output['hidden_states'][-1]

    output = {
      'word_embeds': word_embeds,
      'initial_embeds': initial_embeds,
      'initial_word_embeds': initial_word_embeds,
      'intra_word_mask': intra_word_mask,
      'char_embeds': input_embeds,
      'input_shape': (batch_size, num_word, num_char, hidden_size),
      **inter_word_output,
    }

    return output

  def apply_state(self, state=None):
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
    """ Prediction head used for masked language modeling. """
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.intra_word_encoder.hidden_size, config.vocab_size)
        #self.transform_act_fn = ACT2FN[config.intra_word_encoder.hidden_act] if isinstance(config.intra_word_encoder.hidden_act, str) else config.intra_word_encoder.hidden_act
        #self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        #self.LayerNorm = LayerNorm(config.intra_word_encoder.hidden_size, config.intra_word_encoder.layer_norm_eps, elementwise_affine=True)

        self.residual_word_embedding = getattr(config, 'residual_word_embedding', True)

        intra_word_encoder_config = copy.copy(config.intra_word_encoder)
        intra_word_encoder_config.num_hidden_layers = 1
        self.intra_word_encoder = BertEncoder(intra_word_encoder_config, shared_rel_embeddings=True)

    def forward(self, deberta_output, label_index=None, rel_embeddings=None):
        batch_size, num_word, num_char, hidden_size = deberta_output['input_shape']
        word_embeds = deberta_output['word_embeds']
        initial_embeds = deberta_output['initial_embeds']
        intra_word_mask = deberta_output['intra_word_mask']

        word_embeds = word_embeds.reshape(batch_size * num_word, 1, hidden_size)

        if self.residual_word_embedding:
          # residual connection between initial word embeddings and contextual word embeddings as mentioned in the paper (section A.3)
          initial_word_embeds = deberta_output['initial_word_embeds']
          word_embeds += initial_word_embeds.unsqueeze(1)

        # concatenate to restore the character-level token sequence
        char_embeds = torch.cat([word_embeds, initial_embeds[:,1:,:]], dim=1)
        intra_word_output = self.intra_word_encoder(char_embeds, intra_word_mask, output_all_encoded_layers=False, return_att=False, relative_embeddings=rel_embeddings)
        hidden_states = intra_word_output['hidden_states'][-1]

        if label_index is not None:
          hidden_states = hidden_states.view(-1, hidden_states.size(-1))
          hidden_states = hidden_states.index_select(0, label_index)

        # TODO: Experiment with tied weights (like in regular BERT)
        char_logits = self.dense(hidden_states)
        #char_logits = self.transform_act_fn(char_logits)

        if label_index is None:
          char_logits = char_logits.reshape(batch_size, num_word * num_char, -1)

        # TODO: LayerNorm?

        return char_logits


class CharToWord_LMMaskPredictionHead(torch.nn.Module):
    """ Prediction head used for replaced token detection. """
    def __init__(self, config):
        super().__init__()
        self.layer_norm = LayerNorm(config.intra_word_encoder.hidden_size, config.intra_word_encoder.layer_norm_eps, elementwise_affine=True)
        self.classifier = torch.nn.Linear(config.intra_word_encoder.hidden_size, 1)

        self.residual_word_embedding = getattr(config, 'residual_word_embedding', True)

        intra_word_encoder_config = copy.copy(config.intra_word_encoder)
        intra_word_encoder_config.num_hidden_layers = 1
        self.intra_word_encoder = BertEncoder(intra_word_encoder_config, shared_rel_embeddings=True)

    def forward(self, deberta_output, rel_embeddings=None):
        batch_size, num_word, num_char, hidden_size = deberta_output['input_shape']
        word_embeds = deberta_output['word_embeds']
        initial_embeds = deberta_output['initial_embeds']
        intra_word_mask = deberta_output['intra_word_mask']

        word_embeds = word_embeds.reshape(batch_size * num_word, 1, hidden_size)

        if self.residual_word_embedding:
          # residual connection between initial word embeddings and contextual word embeddings as mentioned in the paper (section A.3)
          initial_word_embeds = deberta_output['initial_word_embeds']
          word_embeds += initial_word_embeds.unsqueeze(1)

        # concatenate to restore the character-level token sequence
        char_embeds = torch.cat([word_embeds, initial_embeds[:,1:,:]], dim=1)
        intra_word_output = self.intra_word_encoder(char_embeds, intra_word_mask, output_all_encoded_layers=False, return_att=False, relative_embeddings=rel_embeddings)
        hidden_states = intra_word_output['hidden_states'][-1]

        #ctx_states = hidden_states[:,0,:]
        #seq_states = self.layer_norm(ctx_states.unsqueeze(-2) + hidden_states)
        #seq_states = self.dense(seq_states)
        #seq_states = self.transform_act_fn(seq_states)

        logits = self.classifier(hidden_states)
        logits = logits.reshape(batch_size, num_word * num_char, -1)

        return logits
