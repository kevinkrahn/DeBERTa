#!/bin/bash

set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$ROOT_DIR"
cd "$ROOT_DIR"

if [ $# -lt 4 ]; then
	echo "Usage: $0 <task> <config_path> <vocab_path> <output_dir> <train_file> <validation_file>"
	exit 0
fi

task=$1
config=$2
vocab_path=$3
output_dir=$4
train_file=$5
validation_file=$6

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
	--num_train_epochs 1 \
	--max_seq_len $max_seq_length \
	--data_dir "$data_dir" \
	--vocab_path "$vocab_path" \
	--vocab_type $vocab_type \
	--output_dir "$output_dir" \
	--task_name "$task" \
	--dataloader_buffer_size 5 \
	--workers 1 \
	--dump 200 \
	--train_batch_size 4
