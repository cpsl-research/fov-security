#!/usr/bin/env bash

set -e

# dataset and model parameters
BASE_CFG="../../config/segmentation/carla/unet_carla_benign.py"
OUT_DIR="./models/carla_parametric"
CFG_DIR="$OUT_DIR/configs"
MOD_DIR="$OUT_DIR/models"

mkdir -p ${CFG_DIR}
mkdir -p ${MOD_DIR}

# parameterize over model width, model depth, and input resolution
WIDTHS=( 4 8 16 32 )
DEPTHS=( 3 4 5 6 )
RESOLS=( 64 128 256 512 )

# make parametric config files
python make_parametric_configs.py \
    --base_cfg "${BASE_CFG}" \
    --out_dir "${CFG_DIR}" \
    --widths "${WIDTHS[@]}" \
    --depths "${DEPTHS[@]}" \
    --resolutions "${RESOLS[@]}"


# loop over derived config files and run training
printf "Searching for .py files in ${CFG_DIR}"
CFG_FILES=(${CFG_DIR}/*.py)
printf "Found ${#CFG_FILES[@]} files"        # print array length
for i_file in "${!CFG_FILES[@]}"; do
    cfg_file=${CFG_FILES[i_file]}

    if [[ ! -e "$cfg_file" ]]; then continue; fi

    # print status
    n_file=$((i_file+1))
    printf "\n\n\nRUNNING CASE ${n_file} OF ${#CFG_FILES[@]}\n\n\n"

    # get directory to save the model
    mod_dir="${cfg_file/"/configs"/"/models"}"
    mod_dir="${mod_dir/".py"/""}"

    # check if it has already been trained
    if [ -d "$mod_dir" ]; then
    printf "Case already exists! Moving on."
        continue
    fi
    echo "$mod_dir"

    # run training for this model
    echo "Training model from config at ${cfg_file}, saving at ${MOD_DIR}"
    python train_segmentation_model.py \
        --config "${cfg_file}" \
        --out_dir "${MOD_DIR}" \
        --gpu 1
done