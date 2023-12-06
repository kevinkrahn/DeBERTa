from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from ...deberta import CharToWord_DeBERTa, CharToWord_LMPredictionHead
from ...deberta.nnmodule import NNModule

__all__ = ['CharToWord_MaskedLanguageModel']

class CharToWord_MaskedLanguageModel(NNModule):
  """ Masked language model with DeBERTa
  """
  def __init__(self, config, *wargs, **kwargs):
    super().__init__(config)
    self.deberta = CharToWord_DeBERTa(config)
    self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
    self.position_buckets = getattr(config, 'position_buckets', -1)
    if self.max_relative_positions < 1:
      self.max_relative_positions = config.max_position_embeddings
    self.lm_predictions = CharToWord_LMPredictionHead(self.deberta.config)
    self.apply(self.init_weights)

  def forward(self, input_ids, char_input_mask, word_input_mask=None, labels=None, position_ids=None):
    device = list(self.parameters())[0].device
    input_ids = input_ids.to(device)
    char_input_mask = char_input_mask.to(device)
    lm_labels = labels.to(device) if labels is not None else None
    if position_ids is not None:
      position_ids = position_ids.to(device)

    encoder_output = self.deberta(input_ids, char_input_mask, output_all_encoded_layers=False, position_ids=position_ids)
    label_index = (lm_labels > 0).nonzero().reshape(-1)
    lm_logits = self.lm_predictions(encoder_output, label_index)
    #lm_logits = lm_logits.view(-1, lm_logits.size(-1))
    lm_labels = lm_labels.view(-1)
    lm_labels = lm_labels.index_select(0, label_index)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    lm_loss = loss_fct(lm_logits, lm_labels)

    return {
      'logits' : lm_logits,
      'labels' : lm_labels,
      'loss' : lm_loss.float(),
    }