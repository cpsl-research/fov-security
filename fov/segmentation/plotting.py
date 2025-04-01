import numpy as np


fov_bev_classes = {
    "invisible": 0,
    "visible": 1,
}

fov_bev_pallete = {
    0: [128, 64, 128],
    1: [10, 120, 232],
}


fov_bev_pallete_matrix = np.zeros((0, 3), dtype=np.uint8)
for i in range(len(fov_bev_pallete)):
    fov_bev_pallete_matrix = np.concatenate(
        (
            fov_bev_pallete_matrix,
            np.array(fov_bev_pallete[i], dtype=np.uint8)[:, None].T,
        ),
        axis=0,
    )
