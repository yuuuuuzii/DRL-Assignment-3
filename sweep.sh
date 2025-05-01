#!/usr/bin/env bash
set -e

OUTPUT="output.txt"

echo "sigma_init" > ${OUTPUT}

for sigma_init in 10 14 20; do
  for tau in 0.00005 0.00003 0.00008; do
    for lr in 0.0008 0.001 0.0004; do
      echo "Testing sigma_init=${sigma_init}, tau=${tau}, lr=${lr}..."
      python train.py \
        --sigma_init ${sigma_init} \
        --tau ${tau} \
        --lr ${lr}
    done
  done
done
