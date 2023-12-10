import torch
from ...deberta import *

__all__ = ['CharToWord_ReplacedTokenDetectionModel']

class CharToWord_ReplacedTokenDetectionModel(NNModule):
  """ RTD with CharToWord_DeBERTa
  """
  def __init__(self, config, *wargs, **kwargs):
    super().__init__(config)
    self.deberta = CharToWord_DeBERTa(config)

    #self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
    #self.position_buckets = getattr(config, 'position_buckets', -1)
    #if self.max_relative_positions <1:
    #  self.max_relative_positions = config.max_position_embeddings
    self.mask_predictions = CharToWord_LMMaskPredictionHead(self.deberta.config)
    self.apply(self.init_weights)

  def forward(self, input_ids, char_input_mask, word_input_mask, char_position_ids=None, word_position_ids=None, labels=None):
    device = list(self.parameters())[0].device

    char_input_mask = char_input_mask.to(device)
    labels = labels.to(device) if labels is not None else None
    if char_position_ids is not None:
      char_position_ids = char_position_ids.to(device)
    if word_position_ids is not None:
      word_position_ids = word_position_ids.to(device)

    encoder_output = self.deberta(input_ids, char_input_mask,
      word_input_mask, char_position_ids, word_position_ids,
      output_all_encoded_layers=False)
    logits = self.mask_predictions(encoder_output, self.deberta.intra_word_encoder.get_rel_embedding())

    #mask_loss = torch.tensor(0).to(logits).float()
    mask_logits = logits.view(-1)
    input_mask = char_input_mask.view(-1).to(mask_logits)
    input_idx = (input_mask > 0).nonzero().view(-1)
    mask_labels = ((labels > 0) & (labels != input_ids.view(labels.shape))).view(-1)
    mask_labels = torch.gather(mask_labels.to(mask_logits), 0, input_idx)
    mask_logits = torch.gather(mask_logits, 0, input_idx).float()
    mask_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    mask_loss = mask_loss_fn(mask_logits, mask_labels)
    return {
      'logits' : mask_logits,
      'labels' : mask_labels,
      'loss' : mask_loss.float(),
    }
