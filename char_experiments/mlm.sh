#!/bin/bash

set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$ROOT_DIR"
cd "$ROOT_DIR"

if [ $# -lt 4 ]; then
	echo "Usage: $0 <config_path> <model_dir> <train_file> <validation_file>"
	exit 0
fi

max_seq_length=512
config=$1
model_dir=$2
data_dir="$model_dir/data"
train_file="$3"
validation_file="$4"

mkdir -p "$data_dir"

python "$SCRIPT_DIR/prepare_data.py" --train "$train_file" --valid "$validation_file" --output "$data_dir" --max_seq_length $max_seq_length

python -m DeBERTa.apps.run --model_config "$config"  \
	--do_train \
	--num_training_steps 1000000 \
	--max_seq_len $max_seq_length \
	--dump 2 \
	--task_name Char_MLM \
	--data_dir "$data_dir" \
	--vocab_path "$data_dir/vocab.json" \
	--vocab_type char \
	--train_batch_size 32 \
	--output_dir "$model_dir"
