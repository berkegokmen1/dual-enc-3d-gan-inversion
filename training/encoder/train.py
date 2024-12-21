import os
import json
import sys
import pprint
import torch

sys.path.append(".")
sys.path.append("..")

from training.encoder.options import TrainOptions
from training.encoder.coach import Coach


def main():

    opts = TrainOptions().parse()

    if os.path.exists(opts.exp_dir) and not opts.debug:
        raise Exception("Oops... {} already exists".format(opts.exp_dir))

    os.makedirs(opts.exp_dir, exist_ok=True)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    with open(os.path.join(opts.exp_dir, "opt.json"), "w") as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    coach = Coach(opts)
    coach.train()


if __name__ == "__main__":
    main()
