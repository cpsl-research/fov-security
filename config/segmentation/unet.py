
dataset_path = "/data/shared/fov/fov_bev_segmentation"

model = dict(
    type="UNetBinary",
    n_channels=1,
    n_classes=1,
    p_dropout=0,
    first_layer_channels=8,
    bilinear=False,
)