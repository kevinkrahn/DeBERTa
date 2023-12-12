#!/bin/bash

set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$ROOT_DIR"
cd "$ROOT_DIR"

if [ $# -lt 4 ]; then
	echo "Usage: $0 <task> <token_format> <config_path> <vocab_path> <output_dir> <train_file> <validation_file>"
	echo "       Task: MLM, RTD"
	echo "       Token format: subword, char, char_to_word"
	exit 0
fi

task=$1
token_format=$2
config=$3
vocab_path=$4
output_dir=$5
train_file=$6
validation_file=$7

max_seq_length=256
max_word_length=16
vocab_type=spm
data_dir="$output_dir/data"

mkdir -p "$data_dir"

python "$SCRIPT_DIR/prepare_data.py" \
	--train_file "$train_file" \
	--valid_file "$validation_file" \
	--vocab_path "$vocab_path" \
	--vocab_type $vocab_type \
	--output_dir "$data_dir" \
	--max_seq_length $max_seq_length \
	--max_word_length $max_word_length \
	--token_format "$token_format"

python -m DeBERTa.apps.run \
	--model_config "$config"  \
	--task_name "$task" \
	--do_train \
	--max_seq_len $max_seq_length \
	--max_word_length $max_word_length \
	--data_dir "$data_dir" \
	--vocab_path "$vocab_path" \
	--vocab_type $vocab_type \
	--output_dir "$output_dir" \
	--token_format "$token_format" \
	--num_train_epochs 1 \
	--learning_rate 1e-4 \
	--train_batch_size 2 \
	--dataloader_buffer_size 5 \
	--workers 1 \
	--fp16 true \
	--dump 1000 \
	--log_steps 250 \
