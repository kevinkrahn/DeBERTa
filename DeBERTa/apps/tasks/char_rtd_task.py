from collections import OrderedDict,defaultdict
import copy
import numpy as np
import os
import sklearn.metrics

import random
import torch
import shutil
from torch.utils.data import DataLoader
from .metrics import *
from .task import EvalData, Task
from .task_registry import register_task
from ...utils import xtqdm as tqdm
from ...data import DynamicDataset
from ...deberta import NNModule
from ...utils import get_logger,boolean_string
from ...training import DistributedTrainer, batch_to
from ...data import DistributedBatchSampler, SequentialSampler, BatchSampler, AsyncDataLoader
from ..models import MaskedLanguageModel, ReplacedTokenDetectionModel
from .char_mlm_task import NGramMaskGenerator
from .._utils import merge_distributed

logger = get_logger()

__all__ = ["Char_RTDTask"]

class RTDModel(NNModule):
  def __init__(self, config, tokenizer=None, *wargs, **kwargs):
    super().__init__(config)
    gen_config = config.generator
    disc_config = config.discriminator
    self.config = config

    if tokenizer and gen_config.vocab_size != len(tokenizer.vocab):
      gen_config.vocab_size = len(tokenizer.vocab)
    if tokenizer and disc_config.vocab_size != len(tokenizer.vocab):
      disc_config.vocab_size = len(tokenizer.vocab)

    self.generator = MaskedLanguageModel(gen_config)
    self.discriminator = ReplacedTokenDetectionModel(disc_config)

    self.generator._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.discriminator._register_load_state_dict_pre_hook(self._pre_load_hook)

    self.share_embedding = getattr(config, 'embedding_sharing', "none").lower()
    if self.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
      word_bias = torch.zeros_like(self.discriminator.deberta.embeddings.word_embeddings.weight)
      word_bias = torch.nn.Parameter(word_bias)
      position_bias = torch.zeros_like(self.discriminator.deberta.embeddings.position_embeddings.weight)
      position_bias = torch.nn.Parameter(position_bias)
      delattr(self.discriminator.deberta.embeddings.word_embeddings, 'weight')
      self.discriminator.deberta.embeddings.word_embeddings.register_parameter('_weight', word_bias)
      delattr(self.discriminator.deberta.embeddings.position_embeddings, 'weight')
      self.discriminator.deberta.embeddings.position_embeddings.register_parameter('_weight', position_bias)
    self.register_discriminator_fw_hook()

  def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
      missing_keys, unexpected_keys, error_msgs):
    bert_prefix = prefix + 'bert.'
    deberta_prefix = prefix + 'deberta.'
    for k in list(state_dict.keys()):
      if k.startswith(bert_prefix):
        nk = deberta_prefix + k[len(bert_prefix):]
        value = state_dict[k]
        del state_dict[k]
        state_dict[nk] = value

  def forward(self, **kwargs):
    return self.generator_fw(**kwargs)

  def discriminator_fw(self, **kwargs):
    return self.discriminator(**kwargs)

  def generator_fw(self, **kwargs):
    return self.generator(**kwargs)

  def topk_sampling(self, logits, topk = 1, start=0, temp=1):
    top_p = torch.nn.functional.softmax(logits/temp, dim=-1)
    topk = max(1, topk)
    next_tokens = torch.multinomial(top_p, topk)
    return next_tokens, top_p
  
  def make_electra_data(self, input_data, temp=1, rand=None):
    new_data = input_data.copy()
    if rand is None:
      rand = random
    gen = self.generator_fw(**new_data)
    lm_logits = gen['logits']
    lm_labels = input_data['labels']
    lm_loss = gen['loss']
    mask_index = (lm_labels.view(-1)>0).nonzero().view(-1)
    gen_pred = torch.argmax(lm_logits, dim=1).detach().cpu().numpy()
    topk_labels, top_p = self.topk_sampling(lm_logits, topk=1, temp=temp)
    
    top_ids = torch.zeros_like(lm_labels.view(-1))
    top_ids.scatter_(index=mask_index, src=topk_labels.view(-1).int(), dim=-1)
    top_ids = top_ids.view(lm_labels.size())
    new_ids = torch.where(lm_labels>0, top_ids, input_data['input_ids'])
    new_data['input_ids'] = new_ids.detach()
    return new_data, lm_loss, gen

  def register_discriminator_fw_hook(self, *wargs):
    def fw_hook(module, *inputs):
      if self.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
        g_w_ebd = self.generator.deberta.embeddings.word_embeddings
        d_w_ebd = self.discriminator.deberta.embeddings.word_embeddings
        self._set_param(d_w_ebd, 'weight', g_w_ebd.weight.detach() + d_w_ebd._weight)

        g_p_ebd = self.generator.deberta.embeddings.position_embeddings
        d_p_ebd = self.discriminator.deberta.embeddings.position_embeddings
        self._set_param(d_p_ebd, 'weight', g_p_ebd.weight.detach() + d_p_ebd._weight)
      elif self.share_embedding == 'es': # vallina embedding sharing
        g_w_ebd = self.generator.deberta.embeddings.word_embeddings
        d_w_ebd = self.discriminator.deberta.embeddings.word_embeddings
        self._set_param(d_w_ebd, 'weight', g_w_ebd.weight)

        g_p_ebd = self.generator.deberta.embeddings.position_embeddings
        d_p_ebd = self.discriminator.deberta.embeddings.position_embeddings
        self._set_param(d_p_ebd, 'weight', g_p_ebd.weight)
      return None
    self.discriminator.register_forward_pre_hook(fw_hook)

  @staticmethod
  def _set_param(module, param_name, value):
    if hasattr(module, param_name):
      delattr(module, param_name)
    module.register_buffer(param_name, value)

@register_task(name="Char_RTD", desc="Replaced token detection pretraining task")
class Char_RTDTask(Task):
  def __init__(self, data_dir, tokenizer, args, **kwargs):
    super().__init__(tokenizer, args, **kwargs)
    self.data_dir = data_dir
    self.mask_gen = NGramMaskGenerator(tokenizer, max_gram=1, keep_prob=0, mask_prob=1, max_seq_len=args.max_seq_length)

  def train_data(self, max_seq_len=512, **kwargs):
    examples = self.load_data(os.path.join(self.data_dir, 'train.txt'))
    dataset_size = len(examples) if self.args.num_training_steps is None else self.args.num_training_steps*self.args.train_batch_size
    return DynamicDataset(examples, feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=self.mask_gen), \
dataset_size=dataset_size, shuffle=True, **kwargs)

  def get_labels(self):
    return list(self.tokenizer.vocab.values())

  def eval_data(self, max_seq_len=512, **kwargs):
    ds = [self._data('dev', 'valid.txt', 'dev')]
   
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

  def get_model_class_fn(self):
    def partial_class(*wargs, **kwargs):
      model = RTDModel.load_model(*wargs, **kwargs)
      if self.args.init_generator is not None:
        logger.info(f'Load generator from {self.args.init_generator}')
        generator = torch.load(self.args.init_generator, map_location='cpu')
        missing_keys, unexpected_keys = model.generator.load_state_dict(generator, strict=False)
        if missing_keys and (len(missing_keys) > 0):
          logger.warning(f'Load generator with missing keys: {missing_keys}')
        if unexpected_keys and (len(unexpected_keys) > 0):
          logger.warning(f'Load generator with unexptected keys: {unexpected_keys}')
      if self.args.init_discriminator is not None:
        logger.info(f'Load discriminator from {self.args.init_discriminator}')
        discriminator = torch.load(self.args.init_discriminator, map_location='cpu')
        missing_keys, unexpected_keys = model.discriminator.load_state_dict(discriminator, strict=False)
        if missing_keys and (len(missing_keys) > 0):
          logger.warning(f'Load discriminator with missing keys: {missing_keys}')
        if unexpected_keys and (len(unexpected_keys) > 0):
          logger.warning(f'Load discriminator with unexptected keys: {unexpected_keys}')
      return model
    return partial_class

  def get_train_fn(self, args, model):
    self.rtd_model = model
    def train_fn(args, model, device, data_fn, eval_fn, loss_fn):
      if args.decoupled_training:
        gen_args = copy.deepcopy(args)
        gen_args.checkpoint_dir = os.path.join(gen_args.output_dir, 'generator')
        os.makedirs(gen_args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(gen_args.checkpoint_dir, 'model_config.json'), 'w') as fs:
          fs.write(model.config.generator.to_json_string() + '\n')
        shutil.copy(args.vocab_path, gen_args.checkpoint_dir)
        loss_fn = self.get_decoupled_loss_fn(args, model, data_fn, device, args.num_training_steps)
        trainer = DistributedTrainer(gen_args, gen_args.output_dir, model.generator, device, data_fn, loss_fn = loss_fn, eval_fn = eval_fn, dump_interval = args.dump_interval, name='G')
      else:
        trainer = DistributedTrainer(args, args.output_dir, model, device, data_fn, loss_fn = loss_fn, eval_fn = eval_fn, dump_interval = args.dump_interval)
      trainer.train()
    return train_fn

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
        mlm_eval_loss = 0
        rtd_eval_loss = 0
        mlm_predicts = []
        mlm_labels = []
        rtd_predicts = []
        rtd_labels = []
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(AsyncDataLoader(eval_dataloader), ncols=80, desc='Evaluating: {}'.format(prefix), disable=no_tqdm):
          batch = batch_to(batch, device)
          with torch.no_grad():
            #output = model(**batch)
            new_data, _, gen_output = self.rtd_model.make_electra_data(batch)
            dis_output = self.rtd_model.discriminator_fw(**new_data)

          # MLM stats
          if 'labels' in gen_output:
            label_ids = gen_output['labels'].detach().to(device)
          else:
            label_ids = batch['labels'].to(device)
          mlm_labels.append(label_ids)
          logits = gen_output['logits'].detach().argmax(dim=-1)
          mlm_predicts.append(logits)
          mlm_eval_loss += gen_output['loss'].detach().mean()

          # RTD stats
          rtd_predicts.append((dis_output['logits'].detach() > 0.5).int())
          rtd_labels.append(dis_output['labels'].to(device))
          rtd_eval_loss += dis_output['loss'].detach().mean()

          input_ids = batch['input_ids']
          nb_eval_examples += input_ids.size(0)
          nb_eval_steps += 1
    
        mlm_eval_loss = mlm_eval_loss / nb_eval_steps
        mlm_predicts = merge_distributed(mlm_predicts)
        mlm_labels = merge_distributed(mlm_labels)

        rtd_eval_loss = rtd_eval_loss / nb_eval_steps
        rtd_predicts = merge_distributed(rtd_predicts)
        rtd_labels = merge_distributed(rtd_labels)

        mlm_results = OrderedDict()
        metrics = eval_item.metrics_fn(mlm_predicts.numpy(), mlm_labels.numpy())
        mlm_results.update(metrics)
        mlm_results['perplexity'] = torch.exp(mlm_eval_loss).item()
        critial_metrics = set(metrics.keys()) if eval_item.critial_metrics is None or len(eval_item.critial_metrics)==0 else eval_item.critial_metrics
        eval_metric = np.mean([v for k,v in metrics.items() if  k in critial_metrics])
        mlm_results['eval_loss'] = mlm_eval_loss.item()
        mlm_results['eval_metric'] = eval_metric
        mlm_results['eval_samples'] = len(mlm_labels)

        rtd_results = OrderedDict()
        rtd_acc = (rtd_predicts == rtd_labels).sum() / len(rtd_labels)
        rtd_f1 = sklearn.metrics.f1_score(rtd_labels, rtd_predicts)
        #rtd_results['perplexity'] = torch.exp(rtd_eval_loss).item()
        rtd_results['eval_loss'] = rtd_eval_loss.item()
        rtd_results['rtd_accuracy'] = rtd_acc.item()
        rtd_results['rtd_f1'] = rtd_f1
        rtd_results['eval_samples'] = len(rtd_labels)

        if args.rank <= 0:
          logger.info("***** Generator (MLM) Eval results-{}-{} *****".format(name, prefix))
          for key in sorted(mlm_results.keys()):
            logger.info("  %s = %s", key, str(mlm_results[key]))

          logger.info("***** Discriminator (MLM) Eval results-{}-{} *****".format(name, prefix))
          for key in sorted(rtd_results.keys()):
            logger.info("  %s = %s", key, str(rtd_results[key]))

        eval_results[name] = (eval_metric, mlm_predicts, mlm_labels)

      return eval_results
    return eval_fn

  def get_decoupled_loss_fn(self, args, model, data_fn, device, num_training_steps):
    rand = random.Random(0)
  
    def eval_fn(trainer, model, device, tag):
      return 0
  
    def d_loss_fn(trainer, model, data):
      disc = model(**data)
      rtd_loss = disc['loss']
      loss = args.rtd_lambda*rtd_loss.mean()
      return loss, data['input_ids'].size(0)
  
    disc_args = copy.deepcopy(args)
    disc_args.checkpoint_dir = os.path.join(disc_args.output_dir, 'discriminator')
    os.makedirs(disc_args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(disc_args.checkpoint_dir, 'model_config.json'), 'w') as fs:
      fs.write(model.config.discriminator.to_json_string() + '\n')
    shutil.copy(args.vocab_path, disc_args.checkpoint_dir)
    if disc_args.discriminator_learning_rate > 0:
      disc_args.learning_rate = disc_args.discriminator_learning_rate
    disc_trainer = DistributedTrainer(disc_args, args.output_dir, model.discriminator, device, data_fn, loss_fn = d_loss_fn, eval_fn = eval_fn, dump_interval = args.dump_interval, name='D')
    disc_trainer.initialize()
  
    def post_g_loss_fn(outputs):
      if outputs is None or len(outputs) == 0:
        return None
      datas = [o['new_data'] for o in outputs]
      new_data = defaultdict(list)
      for d in datas:
        for k in d:
          new_data[k].append(d[k])
      for k in new_data:
        new_data[k] = torch.cat(new_data[k], dim=0)
      disc_trainer._train_step(new_data, 1)
      if disc_trainer.trainer_state.steps % self.args.log_steps == 0:
        disc_trainer.trainer_state.report_state()
  
    def g_loss_fn(trainer, _model, data):
      new_data, mlm_loss, gen_output = model.make_electra_data(data, rand=rand)
      trainer.post_loss_fn = post_g_loss_fn
      loss = mlm_loss.mean()
  
      return {'loss': loss.mean(),
          'batch_size': data['input_ids'].size(0),
          'new_data': new_data}
    return g_loss_fn

  def get_loss_fn(self, args):
    rand = random.Random(0)
    def loss_fn(trainer, model, data):
      train_losses = OrderedDict()
      new_data, mlm_loss, gen_output = model.make_electra_data(data, rand=rand)
      disc = model.discriminator_fw(**new_data)
      rtd_loss = disc['loss']
      loss = mlm_loss.mean() +  args.rtd_lambda*rtd_loss.mean()
      return loss.mean(), data['input_ids'].size(0)
    return loss_fn

  @classmethod
  def add_arguments(cls, parser):
    """Add task specific arguments
      e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
    """
    parser.add_argument('--rtd_lambda', type=float, default=10, help='Weight of RTD loss')
    parser.add_argument('--decoupled_training', type=boolean_string, default=True, help='Whether to use decoupled training')
    parser.add_argument('--num_training_steps', type=int, default=None, help='Maxium pre-training steps')
    parser.add_argument('--discriminator_learning_rate', type=float, default=-1, help='The learning rate of the discriminator')
    parser.add_argument('--init_generator', type=str, default=None, help='The model that used to initialize the generator')
    parser.add_argument('--init_discriminator', type=str, default=None, help='The model that used to initialize the discriminator')

