#!/usr/bin/env bash

set -e

# specify data directories
# CFGFILES=(
#     "carla/unet_carla_benign.py"
#     "nuscenes/unet_nuscenes_benign.py"
#     "ugv/unet_ugv_benign.py"
# )
# OUTDIRS=( "carla" "nuscenes" "ugv" )

# carla only
# CFGFILES=( "carla/unet_carla_benign.py")
# OUTDIRS=( "carla" )

# nuscenes only
# CFGFILES=( "nuscenes/unet_nuscenes_benign.py")
# OUTDIRS=( "nuscenes" )

# ugv only
# CFGFILES=( "ugv/unet_ugv_benign.py")
# OUTDIRS=( "ugv" )

# specify data directories
CFGFILES=(
    #
    "carla/unet_carla_benign.py"
    "nuscenes/unet_nuscenes_benign.py"
    "ugv/unet_ugv_benign.py"
    #
    "carla/unet_carla_benign_mc.py"
    "nuscenes/unet_nuscenes_benign_mc.py"
    "ugv/unet_ugv_benign_mc.py"
    #
    "carla/unet_carla_adversarial.py"
    "nuscenes/unet_nuscenes_adversarial.py"
    "ugv/unet_ugv_adversarial.py"
    #
    "carla/unet_carla_adversarial_mc.py"
    "nuscenes/unet_nuscenes_adversarial_mc.py"
    "ugv/unet_ugv_adversarial_mc.py"
)
OUTDIRS=(
    #
    "carla"
    "nuscenes"
    "ugv"
    #
    "carla"
    "nuscenes"
    "ugv"
    #
    "carla"
    "nuscenes"
    "ugv"
    #
    "carla"
    "nuscenes"
    "ugv"
)



for i in "${!CFGFILES[@]}"
do
    CFGPATH="../../config/segmentation/${CFGFILES[i]}"
    OUTDIR="models/${OUTDIRS[i]}"

    echo "Training from config ${CFGFILES[i]} and saving in ${OUTDIRS[i]} subfolder"
    python train_segmentation_model.py \
        --config "${CFGPATH}" \
        --out_dir "${OUTDIR}" \
        --gpu 1
done
