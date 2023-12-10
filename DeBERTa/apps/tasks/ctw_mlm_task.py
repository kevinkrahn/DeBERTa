from collections import OrderedDict
import numpy as np
import os

import random
import torch
from torch.utils.data import DataLoader
from .metrics import *
from .task import EvalData, Task
from .task_registry import register_task
from ...utils import xtqdm as tqdm
from ...training import batch_to
from ...data import DistributedBatchSampler, SequentialSampler, BatchSampler, AsyncDataLoader
from ...data import DynamicDataset
from ...utils import get_logger
from ..models import CharToWord_MaskedLanguageModel
from .._utils import merge_distributed
from .char_mlm_task import NGramMaskGenerator

logger=get_logger()

__all__ = ["CharToWord_MLMTask"]

@register_task(name="CharToWord_MLM", desc="Masked language model pretraining task")
class CharToWord_MLMTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir
    self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=self.args.max_ngram)

  def train_data(self, max_seq_len=512, **kwargs):
    examples = self.load_data(os.path.join(self.data_dir, 'train.txt'))
    dataset_size = len(examples) if self.args.num_training_steps is None else self.args.num_training_steps*self.args.train_batch_size
    return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), dataset_size=dataset_size, shuffle=True, **kwargs)

  def get_labels(self):
    return list(self.tokenizer.vocab.values())

  def eval_data(self, max_seq_len=512, **kwargs):
    ds = [self._data('dev', 'valid.txt')]
    for d in ds:
      _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), dataset_size=_size, **kwargs)
    return ds

  def _data(self, name, path, ignore_metric=False):
    path = [path] if isinstance(path, str) else path
    examples = []
    for p in path:
      input_src = os.path.join(self.data_dir, p)
      assert os.path.exists(input_src), f"{input_src} doesn't exists"
      examples.extend(self.load_data(input_src))
    predict_fn = self.get_predict_fn()
    return EvalData(name, examples, metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      preds = logits
      acc = (preds == labels).sum() / len(labels)
      metrics = OrderedDict(accuracy=acc)
      return metrics
    return metrics_fn

  def load_data(self, path):
    examples = []
    with open(path, encoding='utf-8') as f:
      for line in f:
        examples.append([int(id) for id in line.strip().split()])
    return examples

  def get_feature_fn(self, max_seq_len=512, mask_gen=None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len=max_seq_len, \
        rng=rng, mask_generator=mask_gen, ext_params=ext_params, **kwargs)
    return _example_to_feature

  def example_to_feature(self, tokenizer, example, max_seq_len, rng=None, mask_generator=None, ext_params=None, **kwargs):
    if not rng:
      rng = random

    tokens = [tokenizer.word_cls_id, tokenizer.cls_id, *example, tokenizer.word_cls_id, tokenizer.sep_id]

    max_word_chars = self.args.max_word_length

    input_ids = []
    char_position_ids = []
    char_input_mask = []

    last_word = []
    num_words = 0
    for i in range(len(tokens)):
      token_id = tokens[i]
      is_end = (i == len(tokens)-1)
      if token_id == self.tokenizer.word_cls_id or is_end:
        if is_end:
          last_word.append(token_id)
        if len(last_word) > 0:
          pad_length = max_word_chars - len(last_word)
          char_position_ids.extend([*range(len(last_word))] + [0]*pad_length)
          char_input_mask.extend([1]*len(last_word) + [0]*pad_length)
          last_word = last_word + [self.tokenizer.pad_id]*pad_length
          input_ids.extend(last_word)
          num_words += 1
        last_word = [token_id]
      else:
        last_word.append(token_id)

    num_pad_words = max_seq_len - num_words
    input_ids.extend([self.tokenizer.pad_id]*max_word_chars*num_pad_words)
    char_position_ids.extend([0]*max_word_chars*num_pad_words)
    char_input_mask.extend([0]*max_word_chars*num_pad_words)
    word_input_mask = [1]*num_words + [0]*num_pad_words
    word_position_ids = [*range(num_words)] + [0]*num_pad_words

    if mask_generator:
      token_ids, labels = mask_generator.mask_tokens(input_ids, rng)

    features = OrderedDict(
      input_ids=torch.tensor(token_ids, dtype=torch.long).reshape(max_seq_len, max_word_chars),
      char_input_mask=torch.tensor(char_input_mask, dtype=torch.long).reshape(max_seq_len, max_word_chars),
      word_input_mask=torch.tensor(word_input_mask, dtype=torch.long),
      char_position_ids=torch.tensor(char_position_ids, dtype=torch.long).reshape(max_seq_len, max_word_chars),
      word_position_ids=torch.tensor(word_position_ids, dtype=torch.long),
      labels=torch.tensor(labels, dtype=torch.long))
    return features

  def get_eval_fn(self):
    def eval_fn(args, model, device, eval_data, prefix=None, tag=None, steps=None):
      # Run prediction for full data
      prefix = f'{tag}_{prefix}' if tag is not None else prefix
      eval_results=OrderedDict()
      eval_metric=0
      no_tqdm = (True if os.getenv('NO_TQDM', '0')!='0' else False) or args.rank>0
      for eval_item in eval_data:
        name = eval_item.name
        eval_sampler = SequentialSampler(len(eval_item.data))
        batch_sampler = BatchSampler(eval_sampler, args.eval_batch_size)
        batch_sampler = DistributedBatchSampler(batch_sampler, rank=args.rank, world_size=args.world_size)
        eval_dataloader = DataLoader(eval_item.data, batch_sampler=batch_sampler, num_workers=args.workers)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicts=[]
        labels=[]
        for batch in tqdm(AsyncDataLoader(eval_dataloader), ncols=80, desc='Evaluating: {}'.format(prefix), disable=no_tqdm):
          batch = batch_to(batch, device)
          with torch.no_grad():
            output = model(**batch)
          logits = output['logits'].detach().argmax(dim=-1)
          tmp_eval_loss = output['loss'].detach()
          if 'labels' in output:
            label_ids = output['labels'].detach().to(device)
          else:
            label_ids = batch['labels'].to(device)
          predicts.append(logits)
          labels.append(label_ids)
          eval_loss += tmp_eval_loss.mean()
          input_ids = batch['input_ids']
          nb_eval_examples += input_ids.size(0)
          nb_eval_steps += 1
    
        eval_loss = eval_loss / nb_eval_steps
        predicts = merge_distributed(predicts)
        labels = merge_distributed(labels)

        result=OrderedDict()
        metrics_fn = eval_item.metrics_fn
        metrics = metrics_fn(predicts.numpy(), labels.numpy())
        result.update(metrics)
        result['perplexity'] = torch.exp(eval_loss).item()
        critial_metrics = set(metrics.keys()) if eval_item.critial_metrics is None or len(eval_item.critial_metrics)==0 else eval_item.critial_metrics
        eval_metric = np.mean([v for k,v in metrics.items() if  k in critial_metrics])
        result['eval_loss'] = eval_loss.item()
        result['eval_metric'] = eval_metric
        result['eval_samples'] = len(labels)
        if args.rank<=0:
          logger.info("***** Eval results-{}-{} *****".format(name, prefix))
          for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        eval_results[name]=(eval_metric, predicts, labels)

      return eval_results
    return eval_fn

  def get_model_class_fn(self):
    def partial_class(*wargs, **kwargs):
      return CharToWord_MaskedLanguageModel.load_model(*wargs, **kwargs)
    return partial_class

  @classmethod
  def add_arguments(cls, parser):
    """Add task specific arguments
      e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
    """
    parser.add_argument('--max_ngram', type=int, default=1, help='Maxium ngram sampling span')
    parser.add_argument('--num_training_steps', type=int, default=None, help='Maxium pre-training steps')
