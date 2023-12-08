from DeBERTa.deberta.tokenizers import tokenizers
import argparse
import conllu

def main(args):
  tokenizer = tokenizers[args.vocab_type](args.vocab_path)
  encode_to_ids(args.train_file, args.output_dir + "/train.txt", args.max_seq_length, args.max_word_length, args.char_to_word, tokenizer)
  encode_to_ids(args.valid_file, args.output_dir + "/valid.txt", args.max_seq_length, args.max_word_length, args.char_to_word, tokenizer)
  if args.test_file:
    encode_to_ids(args.test_file, args.output_dir + "/test.txt", args.max_seq_length, args.max_word_length, args.char_to_word, tokenizer)


def encode_to_ids(input_file, output_file, max_seq_length, max_word_length, char_to_word, tokenizer):
  with open(input_file, encoding='utf-8') as f:
    # Input must be segmented into words because each word is prefixed with [WORD_CLS]
    if input_file.endswith('.conllu'):
      lines = [ [token['form'] for token in sentence] for sentence in conllu.parse(f.read())]
    else:
      # TODO: Use better tokenization if the input is not segmented (e.g. handle punctuation)
      lines = [ l.split() for l in f.readlines() if l.strip() != '' ]

    if char_to_word:
      all_lines = []
      for line in lines:
        words = []
        for word in line:
          # TODO: Split or truncate? Currently, if a word is longer than max_word_length it is split into multiple words
          token_ids = tokenizer.tokenize(word)
          for i in range(0, len(token_ids), max_word_length-1):
            words.append([tokenizer.word_cls_id] + token_ids[i: i+max_word_length-1])
        all_lines.append(words)
    else:
      all_lines = [[[tokenizer.word_cls_id, *tokenizer.tokenize(word)] for word in line] for line in lines]

  # Save the encoded lines to the output file, splitting lines > max_seq_length
  total_lines = 0
  with open(output_file, 'w', encoding='utf-8') as f:
    if char_to_word:
      word_chunk_size = max_seq_length-2
      for line in all_lines:
        idx = 0
        while idx < len(line):
          f.write(' '.join(str(token_id) for word in line[idx:idx+word_chunk_size] for token_id in word) + '\n')
          idx += word_chunk_size
          total_lines += 1
    else:
      for line in all_lines:
        idx = 0
        char_chunk_size = max_seq_length-2
        # Split long lines at word boundaries
        while idx < len(line):
          token_ids = []
          while idx < len(line) and len(token_ids) + len(line[idx]) < char_chunk_size:
            token_ids.extend(line[idx])
            idx += 1
          f.write(' '.join(str(token_id) for token_id in token_ids) + '\n')
          total_lines += 1

  print(f'Saved {total_lines} lines to {output_file}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_file', required=True, help='The train data path')
  parser.add_argument('--valid_file', required=True, help='The validation data path')
  parser.add_argument('--test_file', required=False, help='The test data path')
  parser.add_argument('--vocab_path', required=True, default=None, help='The path to the vocab file')
  parser.add_argument('--vocab_type', required=True, default=None, help='The tokenizer type')
  parser.add_argument('--output_dir', required=True, default=None, help='The output directory')
  parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
  parser.add_argument('--char_to_word', action="store_true", help='Maxium sequence length refers to number of words (for CharToWord model)')
  parser.add_argument('--max_word_length', type=int, default=20, help='Maximum number of chars per word (only applies if --char_to_word is set))')
  args = parser.parse_args()

  main(args)
