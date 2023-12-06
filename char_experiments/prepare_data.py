from DeBERTa.deberta.char_tokenizer import make_vocab, CharTokenizer
import argparse
from tqdm import tqdm

def main(train_file, validation_file, output_dir, max_seq_length, max_word_length, char_to_word, max_eval_words):
  vocab_file = output_dir + '/vocab.json'
  make_vocab(train_file, vocab_file)
  tokenizer = CharTokenizer(vocab_file)
  tokenize_data(train_file, output_dir + "/train.txt", max_seq_length, max_word_length, char_to_word, tokenizer)
  tokenize_data(validation_file, output_dir + "/valid.txt", max_seq_length, max_word_length, char_to_word, tokenizer, max_eval_words)

def tokenize_data(input, output_file, max_seq_length, max_word_length, char_to_word, tokenizer, max_words=-1):
  all_words = []
  with open(input, encoding='utf-8') as f:
    for l in tqdm(f, ncols=80, desc='Loading'):
      # TODO: Replace with better word segmentation (e.g. spacy) or
      # pre-existing word segmentation (e.g. from conllu files)
      words = l.split()
      if char_to_word:
        for word in words:
          # TODO: Split or truncate? Currently, if a word is longer than max_word_length it is split into multiple words
          word_token_ids = tokenizer.tokenize(word)
          for i in range(0, len(word_token_ids), max_word_length-1):
              all_words.append([tokenizer.word_cls_id] + word_token_ids[i: i+max_word_length-1])
      else:
        for word in words:
          word_token_ids = [tokenizer.word_cls_id, *tokenizer.tokenize(word)]
          all_words.append(word_token_ids)

  print(f'Loaded {len(all_words)} words from {input}')

  lines = 0
  with open(output_file, 'w', encoding='utf-8') as f:
    if char_to_word:
      idx = 0
      while idx < len(all_words) and (idx < max_words or max_words < 0):
        f.write(' '.join(str(token_id) for word in all_words[idx:idx+max_seq_length-2] for token_id in word) + '\n')
        idx += max_seq_length
        lines += 1
    else:
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
parser.add_argument('--char_to_word', action="store_true", help='Maxium sequence length refers to number of words (for CharToWord model)')
parser.add_argument('--max_word_length', type=int, default=20, help='Maximum number of chars per word (only applies if --char_to_word is set))')
args = parser.parse_args()

main(args.train, args.valid, args.output, args.max_seq_length, args.max_word_length, args.char_to_word, max_eval_words=200)
