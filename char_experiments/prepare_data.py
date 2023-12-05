from DeBERTa.deberta.char_tokenizer import make_vocab, CharTokenizer
import argparse
from tqdm import tqdm

def main(train_file, validation_file, output_dir, max_seq_length, max_eval_words):
  vocab_file = output_dir + '/vocab.json'
  make_vocab(train_file, vocab_file)
  tokenizer = CharTokenizer(vocab_file)
  tokenize_data(train_file, output_dir + "/train.txt", max_seq_length, tokenizer)
  tokenize_data(validation_file, output_dir + "/valid.txt", max_seq_length, tokenizer, max_eval_words)

def tokenize_data(input, output_file, max_seq_length, tokenizer, max_words=-1):
  all_words = []
  with open(input, encoding='utf-8') as f:
    for l in tqdm(f, ncols=80, desc='Loading'):
      # TODO: replace with better word segmentation (e.g. spacy) or
      # pre-existing word segmentation (e.g. from conllu files)
      words = l.split()
      for word in words:
        word_token_ids = [tokenizer.word_cls_id, *tokenizer.tokenize(word)]
        all_words.append(word_token_ids)
  print(f'Loaded {len(all_words)} tokens from {input}')
  lines = 0
  with open(output_file, 'w', encoding='utf-8') as f:
    idx = 0
    while idx < len(all_words) and (idx < max_words or max_words < 0):
      line = []
      while idx < len(all_words) and len(line) + len(all_words[idx]) < max_seq_length - 2:
        line.extend(all_words[idx])
        idx += 1
      f.write(' '.join(str(token_id) for token_id in line) + '\n')
      #f.write(' '.join(tokenizer.id_to_tokens[token_id] for token_id in line) + ' ' + str(len(line)) + '\n')
      lines += 1

  print(f'Saved {lines} lines to {output_file}')

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help='The train data path')
parser.add_argument('--valid', required=True, help='The validation data path')
parser.add_argument('--test', required=False, help='The test data path')
parser.add_argument('--output', default=None, help='The output directory')
parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
args = parser.parse_args()

main(args.train, args.valid, args.output, args.max_seq_length, max_eval_words=200)
