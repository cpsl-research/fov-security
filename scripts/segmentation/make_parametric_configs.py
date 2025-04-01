import os
import argparse

from avstack.config import Config


def main(args):
    """Make the set of config files for the parametric analysis"""

    base_cfg = Config.fromfile(args.base_cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    # loop over all others
    for width in args.widths:
        base_cfg["model"]["first_layer_channels"] = int(width)
        for depth in args.depths:
            base_cfg["model"]["n_layers"] = int(depth)
            for resol in args.resolutions:
                base_cfg["model_io_size"] = (int(resol), int(resol))

                # save config
                cfg_filepath = os.path.join(args.out_dir, f"width_{width}_depth_{depth}_resolution_{resol}.py")
                base_cfg.dump(cfg_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--widths", nargs="+")
    parser.add_argument("--depths", nargs="+")
    parser.add_argument("--resolutions", nargs="+")
    args = parser.parse_args()

    main(args)