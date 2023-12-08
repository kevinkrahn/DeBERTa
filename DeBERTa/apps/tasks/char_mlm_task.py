from collections import OrderedDict
from bisect import bisect
import math
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
from ..models import MaskedLanguageModel
from .._utils import merge_distributed

logger=get_logger()

__all__ = ["Char_MLMTask"]

class NGramMaskGenerator:
  def __init__(self, tokenizer, mask_lm_prob=0.15, max_seq_len=512, max_preds_per_seq=None, max_gram=1, keep_prob=0.1, mask_prob=0.8, **kwargs):
    self.tokenizer = tokenizer
    self.mask_lm_prob = mask_lm_prob
    self.keep_prob = keep_prob
    self.mask_prob = mask_prob
    assert self.mask_prob+self.keep_prob<=1, f'The prob of using [MASK]({mask_prob}) and the prob of using original token({keep_prob}) should between [0,1]'
    self.max_preds_per_seq = max_preds_per_seq
    if max_preds_per_seq is None:
      self.max_preds_per_seq = math.ceil(max_seq_len*mask_lm_prob/10)*10

    self.max_gram = max(max_gram, 1)
    self.mask_window = int(1/mask_lm_prob) # make ngrams per window sized context
    self.vocab_words = list(set(tokenizer.vocab.values()).difference(tokenizer.special_tokens))

  def mask_tokens(self, tokens, rng, **kwargs):
    ngrams = np.arange(1, self.max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, self.max_gram + 1)
    pvals /= pvals.sum(keepdims=True)

    # Each word is prefixed by a [WORD_CLS] token
    unigrams = []
    last_word = []
    for i in range(len(tokens)):
      token_id = tokens[i]
      if token_id == self.tokenizer.word_cls_id or token_id == self.tokenizer.sep_id:
        if len(last_word) > 0:
          unigrams.append(last_word)
        last_word = []
      elif token_id in self.tokenizer.special_token_indices:
        continue
      else:
        last_word.append(i)

    # Mask random characters instead of entire words if the number of words is small
    if len(unigrams) < 4:
      unigrams = [[i] for i in range(len(tokens)) if tokens[i] not in self.tokenizer.special_token_indices]

    num_to_predict = min(self.max_preds_per_seq, max(1, int(round(len(tokens) * self.mask_lm_prob))))
    offset = 0
    mask_grams = np.array([False]*len(unigrams))
    while offset < len(unigrams):
      n = self._choice(rng, ngrams, p=pvals)
      ctx_size = min(n*self.mask_window, len(unigrams)-offset)
      m = rng.randint(0, ctx_size-1)
      s = offset + m
      e = min(offset+m+n, len(unigrams))
      offset = max(offset+ctx_size, e)
      mask_grams[s:e] = True

    target_labels = [None]*len(tokens)
    w_cnt = 0
    # Mask the chars in the chosen words
    for m,word in zip(mask_grams, unigrams):
      if m:
        for idx in word:
          label = self._mask_token(idx, tokens, rng, self.mask_prob, self.keep_prob)
          target_labels[idx] = label
          w_cnt += 1
        if w_cnt >= num_to_predict:
          break

    target_labels = [token_id if token_id else 0 for token_id in target_labels]
    return tokens, target_labels

  def _choice(self, rng, data, p):
    cul = np.cumsum(p)
    x = rng.random()*cul[-1]
    id = bisect(cul, x)
    return data[id]

  def _mask_token(self, idx, tokens, rng, mask_prob, keep_prob):
    label = tokens[idx]
    rand = rng.random()
    if rand < mask_prob:
      new_label = self.tokenizer.mask_id
    elif rand < mask_prob+keep_prob:
      new_label = label
    else:
      new_label = rng.choice(self.vocab_words)

    tokens[idx] = new_label

    return label

@register_task(name="Char_MLM", desc="Masked language model pretraining task")
class Char_MLMTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir
    self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=self.args.max_ngram)

  def train_data(self, max_seq_len=512, **kwargs):
    examples = self.load_data(os.path.join(self.data_dir, 'train.txt'))
    dataset_size = len(examples) if self.args.num_training_steps is None else self.args.num_training_steps*self.args.train_batch_size
    return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), \
dataset_size = dataset_size, shuffle=True, **kwargs)

  def get_labels(self):
    return list(self.tokenizer.vocab.values())

  def eval_data(self, max_seq_len=512, **kwargs):
    ds = [ self._data('dev', 'valid.txt', 'dev') ]
    for d in ds:
      _size = len(d.data)
      d.data = DynamicDataset(d.data, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), dataset_size = _size, **kwargs)
    return ds

  def test_data(self, max_seq_len=512, **kwargs):
    """See base class."""
    raise NotImplemented('This method is not implemented yet.')

  def _data(self, name, path, type_name='dev', ignore_metric=False):
    if isinstance(path, str):
      path = [path]
    examples = []
    for p in path:
      input_src = os.path.join(self.data_dir, p)
      assert os.path.exists(input_src), f"{input_src} doesn't exists"
      examples.extend(self.load_data(input_src))

    predict_fn = self.get_predict_fn()
    return EvalData(name, examples,
      metrics_fn=self.get_metrics_fn(), predict_fn=predict_fn, ignore_metric=ignore_metric, critial_metrics=['accuracy'])

  def get_metrics_fn(self):
    """Calcuate metrics based on prediction results"""
    def metrics_fn(logits, labels):
      preds = logits
      acc = (preds==labels).sum()/len(labels)
      metrics =  OrderedDict(accuracy= acc)
      return metrics
    return metrics_fn

  def load_data(self, path):
    examples = []
    with open(path, encoding='utf-8') as f:
      for line in f:
        examples.append([int(id) for id in line.strip().split()])
    return examples

  def get_feature_fn(self, max_seq_len = 512, mask_gen = None):
    def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
      return self.example_to_feature(self.tokenizer, example, max_seq_len = max_seq_len, \
        rng = rng, mask_generator = mask_gen, ext_params = ext_params, **kwargs)
    return _example_to_feature

  def example_to_feature(self, tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, **kwargs):
    if not rng:
      rng = random
    assert(len(example) < max_seq_len-2)
    _tokens = [tokenizer.cls_id, *example, tokenizer.sep_id]
    if mask_generator:
      token_ids, lm_labels = mask_generator.mask_tokens(_tokens, rng)

    features = OrderedDict(input_ids=token_ids, position_ids=list(range(len(token_ids))), input_mask=[1]*len(token_ids), labels=lm_labels)
    
    for f in features:
      features[f] = torch.tensor(features[f] + [0]*(max_seq_len - len(token_ids)), dtype=torch.int)
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
      return MaskedLanguageModel.load_model(*wargs, **kwargs)
    return partial_class

  @classmethod
  def add_arguments(cls, parser):
    """Add task specific arguments
      e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
    """
    parser.add_argument('--max_ngram', type=int, default=1, help='Maxium ngram sampling span')
    parser.add_argument('--num_training_steps', type=int, default=None, help='Maxium pre-training steps')
