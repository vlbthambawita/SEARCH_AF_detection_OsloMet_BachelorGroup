#!/bin/bash

DATASETS=("ptbxl" "ecg-arrhythmia" "mit-bih")
SAMPLING_RATES=("100hz" "250hz" "500hz")
MODELS=("cnn1d" "resnet1d")

for dataset in "${DATASETS[@]}"; do
  for rate in "${SAMPLING_RATES[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "======================================"
      echo "Dataset: $dataset | Rate: $rate | Model: $model"
      echo "======================================"

      python train.py \
        --data_path data/$dataset/$rate \
        --model $model
    done
  done
done
