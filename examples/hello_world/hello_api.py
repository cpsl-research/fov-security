import logging
import os

import avapi


dir_path = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    data_dir = os.path.join(dir_path, "..", "..", "data")
    test_dict = {
        "kitti": {
            "active": True,
            "data_dir": os.path.join(data_dir, "KITTI", "object"),
            "factory": avapi.kitti.KittiScenesManager,
        },
        "nuscenes": {
            "active": True,
            "data_dir": os.path.join(data_dir, "nuScenes"),
            "factory": avapi.nuscenes.nuScenesManager,
        },
        "carla": {
            "active": True,
            "data_dir": os.path.join(data_dir, "CARLA"),
            "factory": avapi.carla.CarlaScenesManager,
        },
    }

    for dataset, params in test_dict.items():
        if params["active"]:
            print(f"\nTesting {dataset}...")
            SM = params["factory"](params["data_dir"])
            if len(SM) == 0:
                logging.warning(f"No scenes found for {dataset}!")
            else:
                print(f"Found {len(SM)} scenes!")
        else:
            print(f"\nNot testing {dataset}")
