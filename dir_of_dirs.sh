#!/bin/bash
# script for generating new light dataset

for in_dir in "$1"/*; do
  out_dir="${in_dir}_glare";
  mkdir -p "$out_dir";
  python code/infer_unpaired.py -i  "$in_dir" -o "$out_dir";
  echo "done for ${in_dir}";
done