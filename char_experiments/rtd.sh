#!/bin/bash

set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$ROOT_DIR"
cd "$ROOT_DIR"

if [ $# -lt 4 ]; then
	echo "Usage: $0 <config_path> <vocab_path> <output_dir> <train_file> <validation_file>"
	exit 0
fi

config=$1
vocab_path=$2
output_dir=$3
train_file=$4
validation_file=$5

max_seq_length=512
vocab_type=spm_char
data_dir="$output_dir/data"

mkdir -p "$data_dir"

python "$SCRIPT_DIR/prepare_data.py" \
	--train_file "$train_file" \
	--valid_file "$validation_file" \
	--vocab_path "$vocab_path" \
	--vocab_type $vocab_type \
	--output_dir "$data_dir" \
	--max_seq_length $max_seq_length

python -m DeBERTa.apps.run \
	--model_config "$config"  \
	--do_train \
	--num_training_steps 20 \
	--max_seq_len $max_seq_length \
	--data_dir "$data_dir" \
	--vocab_path "$vocab_path" \
	--vocab_type $vocab_type \
	--output_dir "$output_dir" \
	--task_name Char_RTD \
	--dump 20 \
	--train_batch_size 32
