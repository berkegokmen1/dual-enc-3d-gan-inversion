import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import json
import torch
import random
from dataset.dataset import ImageFolderDataset
import numpy as np
from torch.utils.data import DataLoader, random_split
import argparse
import torchvision

from tqdm import tqdm


class Inference:
    def __init__(self, opts) -> None:
        self.opts = opts

        self.global_step = 0
        torch.backends.cudnn.benchmark = True

        SEED = 2107
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        _, self.test_dataset = self.configure_datasets()
        self.configure_dataloaders(batch_size=1)

    def configure_datasets(self):

        full_dataset = ImageFolderDataset(
            path=self.opts.data_path,
            synth_path=self.opts.synth_data_path,
            load_original_ws=False,
            synth_data_ratio=0,
            use_labels=True,
        )

        train_size, test_size = 1 - 0.05, 0.05
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def configure_dataloaders(self, batch_size):
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )

    def generate_from_images(self):
        torch.cuda.empty_cache()

        dataset_json = {"labels": []}

        for batch_idx, batch in tqdm(enumerate(self.test_dataloader), total=self.opts.num_samples):
            if batch_idx >= self.opts.num_samples:
                print("Finished generating samples.")
                break

            (
                x_resized,
                x,
                camera_param,
                x_mirror_resized,
                x_mirror,
                camera_param_mirror,
                w_original,
                fname,
                synth_data,
            ) = batch

            torchvision.utils.save_image(
                ((x + 1) / 2).clamp(0, 1),
                f"{self.opts.output_dir}/{fname[0]}",
                value_range=(0, 1),
            )
            dataset_json["labels"].append([fname[0], camera_param.numpy().tolist()[0]])

        print("Done creating samples!")

        with open(f"{self.opts.output_dir}/dataset.json", "w") as f:
            json.dump(dataset_json, f)

        print("Saved dataset.json!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained PanoHead model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", required=True)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--data_path", type=str, help="Path to data images.", required=True)
    parser.add_argument("--synth_data_path", type=str, help="Path to syhtn data images.", required=True)

    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    inference = Inference(opts)

    inference.generate_from_images()
