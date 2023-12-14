from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from ...deberta import CharToWord_DeBERTa, CharToWord_LMPredictionHead
from ...deberta.nnmodule import NNModule

__all__ = ['CharToWord_MaskedLanguageModel']

class CharToWord_MaskedLanguageModel(NNModule):
  """ Masked language model with CharToWord_DeBERTa
  """
  def __init__(self, config, *wargs, **kwargs):
    super().__init__(config)
    self.deberta = CharToWord_DeBERTa(config)
    self.lm_predictions = CharToWord_LMPredictionHead(self.deberta.config, self.deberta.char_embeddings)
    self.apply(self.init_weights)

  def forward(self, input_ids, char_input_mask, word_input_mask, char_position_ids=None, word_position_ids=None, labels=None):
    device = list(self.parameters())[0].device
    input_ids = input_ids.to(device)
    char_input_mask = char_input_mask.to(device)
    lm_labels = labels.to(device) if labels is not None else None
    if char_position_ids is not None:
      char_position_ids = char_position_ids.to(device)
    if word_position_ids is not None:
      word_position_ids = word_position_ids.to(device)

    encoder_output = self.deberta(input_ids, char_input_mask,
      word_input_mask, char_position_ids, word_position_ids,
      output_all_encoded_layers=False)
    lm_labels = lm_labels.view(-1)
    label_index = (lm_labels > 0).nonzero().view(-1)
    lm_logits = self.lm_predictions(encoder_output, label_index, self.deberta.intra_word_encoder.get_rel_embedding())
    lm_labels = lm_labels.index_select(0, label_index)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    lm_loss = loss_fct(lm_logits, lm_labels)

    return {
      'logits' : lm_logits,
      'labels' : lm_labels,
      'loss' : lm_loss.float(),
    }
