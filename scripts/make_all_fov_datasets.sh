#!/usr/bin/env bash

set -e


# all datasets
DATANAMES=(
    # "carla"
    "nuscenes"
    # "ugv"
    # "carla_adversarial"
    "nuscenes_adversarial"
    # "ugv_adversarial"
)

# DATANAMES=(
#     # "carla"
#     # "nuscenes"
#     # "ugv"
#     "carla_adversarial"
#     "nuscenes_adversarial"
#     "ugv_adversarial"
# )

# # only carla
# DATANAMES=( "carla" )

# only nuscenes
# DATANAMES=( "nuscenes" )

# only ugv
# DATANAMES=( "ugv" )

# loop over the data
for i in "${!DATANAMES[@]}"
do
    echo "Making dataset ${DATANAMES[i]}"
    CFG="../config/segmentation/__base__/datasets/${DATANAMES[i]}.py" 
    
    # make benign dataset
    python make_fov_datasets.py \
        --dataset_config ${CFG} \
        --seed 0
done
