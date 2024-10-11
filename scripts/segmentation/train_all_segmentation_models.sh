#!/usr/bin/env bash

set -e

MODELDIR=${1:-../config/segmentation}

for cfg_file in ${MODELDIR}/*.py; do
    echo "Running on config ${cfg_file}"
    python train_segmentation_model.py \
        --config "${cfg_file}"
done