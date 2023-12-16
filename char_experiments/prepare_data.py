from DeBERTa.deberta.tokenizers import tokenizers
import argparse
import conllu

def encode(input_file, output_file, max_seq_length, max_word_length, token_format, tokenizer, split_long_words=True):
  print(f'Loading {input_file}...')
  with open(input_file, encoding='utf-8') as f:
    # Input must be segmented into words because each word is prefixed with [WORD_CLS]
    if input_file.endswith('.conllu'):
      lines = [ [token['form'] for token in sentence] for sentence in conllu.parse(f.read())]
    else:
      # TODO: Use better tokenization if the input is not segmented (e.g. handle punctuation)
      lines = [ l.split() for l in f.readlines() if l.strip() != '' ]

    split_words = 0
    if token_format == 'char_to_word':
      all_lines = []
      for line in lines:
        words = []
        for word in line:
          tokens = tokenizer.tokenize(word)
          if split_long_words:
            for i in range(0, len(tokens), max_word_length-1):
              words.append(['[WORD_CLS]'] + tokens[i: i+max_word_length-1])
          else:
              words.append(['[WORD_CLS]'] + tokens[0: max_word_length-1])
        all_lines.append(words)
        split_words += len(words) - len(line)
    elif token_format == 'char':
      all_lines = [[['[WORD_CLS]', *tokenizer.tokenize(word)] for word in line] for line in lines]
    elif token_format == 'subword':
      all_lines = [[tokenizer.tokenize(word) for word in line] for line in lines]
    else:
      raise ValueError(f'Invalid token_format: {token_format}')

  # Save the encoded lines to the output file, splitting lines > max_seq_length
  total_lines = 0
  with open(output_file, 'w', encoding='utf-8') as f:
    if token_format == 'char_to_word':
      word_chunk_size = max_seq_length-2
      for line in all_lines:
        idx = 0
        while idx < len(line):
          f.write(' '.join(token for word in line[idx:idx+word_chunk_size] for token in word) + '\n')
          idx += word_chunk_size
          total_lines += 1
    else:
      for line in all_lines:
        idx = 0
        char_chunk_size = max_seq_length-2
        # Split long lines at word boundaries
        while idx < len(line):
          tokens = []
          while idx < len(line) and len(tokens) + len(line[idx]) < char_chunk_size:
            tokens.extend(line[idx])
            idx += 1
          f.write(' '.join(token for token in tokens) + '\n')
          total_lines += 1

  split_lines = total_lines - len(lines)

  print(f'Saved {total_lines} lines to {output_file}')
  print(f'Split {split_lines} lines into multiple lines of max_seq_length={max_seq_length}')
  if split_words > 0:
    print(f'Split {split_words} words into multiple words of max_word_length={max_word_length}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_file', required=True, help='The train data path')
  parser.add_argument('--valid_file', required=True, help='The validation data path')
  parser.add_argument('--test_file', required=False, help='The test data path')
  parser.add_argument('--vocab_path', required=True, default=None, help='The path to the vocab file')
  parser.add_argument('--vocab_type', default='spm', help='The tokenizer type')
  parser.add_argument('--output_dir', required=True, default=None, help='The output directory')
  parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
  parser.add_argument('--token_format', required=True, type=str, help='How to store the tokens in the output files. Options: char, word, char_to_word')
  parser.add_argument('--max_word_length', type=int, default=20, help='Maximum number of chars per word (only applies if --char_to_word is set))')
  args = parser.parse_args()

  tokenizer = tokenizers[args.vocab_type](args.vocab_path)
  encode(args.train_file, args.output_dir + "/train.txt", args.max_seq_length, args.max_word_length, args.token_format, tokenizer)
  encode(args.valid_file, args.output_dir + "/valid.txt", args.max_seq_length, args.max_word_length, args.token_format, tokenizer)
  if args.test_file:
    encode(args.test_file, args.output_dir + "/test.txt", args.max_seq_length, args.max_word_length, args.token_format, tokenizer)
