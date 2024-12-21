import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import json
import legacy
import argparse
import numpy as np
import PIL.Image
from tqdm import tqdm

import dnnlib.util
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from face_parsing_pytorch.model import BiSeNet

bisenet_attribute_chs = {
    "mouth": [11, 12, 13],
    "face_wo_neck": np.arange(1, 16),
    "face": np.arange(1, 17),
    "glasses": [6],
    "hair": [17],
    "brows": [2, 3],
    "eyes": [4, 5],
    "nose": [10],
}

yaw_range_back = [(-np.pi / 2) + (np.pi / 2), (-np.pi / 2) - (np.pi / 2)]
yaw_range_front = [(np.pi / 2) - (np.pi / 2), (np.pi / 2) + (np.pi / 2)]
pitch_range = [np.pi / 2 - (np.pi / 3), np.pi / 2 + (np.pi / 3)]

hair_threshold_back = 0.40  # gut feeling
hair_threshold_front = 0.20  # gut feeling


def gen_samples(args):
    camera_params_dict = {"labels": []}

    device = torch.device(args.device)

    print('Loading networks from "%s"...' % args.network_pkl)

    with dnnlib.util.open_url(args.network_pkl) as f:
        pickle_dict = legacy.load_network_pkl(f)
        G = pickle_dict["G_ema"].to(device)

    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    mask_net = BiSeNet(n_classes=19).to(device).requires_grad_(False)
    mask_gen_ckpt = torch.load(args.bisenet_ckpt, map_location=device)
    mask_net.load_state_dict(mask_gen_ckpt)
    print(f"Loaded {args.bisenet_ckpt}!")
    del mask_gen_ckpt
    mask_net.eval()

    pose_cond_rad = args.pose_cond / 180 * np.pi

    intrinsics = FOV_to_intrinsics(args.fov_deg, device=device)

    np.random.seed(0)

    seeds = np.random.randint(0, 10**6, args.num_samples * 2)

    created_samples = 0

    pbar = tqdm(total=args.num_samples, desc="Generating samples", unit="samples")

    for seed_idx, seed in enumerate(seeds):
        if created_samples >= args.num_samples:
            break

        try:
            # print("Generating image for seed %d (%d/%d) ..." % (seed, created_samples + 1, args.num_samples))

            cam_pivot = torch.tensor([0, 0, 0], device=device)
            cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)

            conditioning_cam2world_pose = LookAtPoseSampler.sample(
                pose_cond_rad, np.pi / 2, cam_pivot, radius=cam_radius, device=device
            )
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            yaw_range = yaw_range_back if seed_idx % 2 == 0 else yaw_range_front

            cam2world_pose = LookAtPoseSampler.sample(
                np.random.uniform(yaw_range[0], yaw_range[1]),
                np.random.uniform(pitch_range[0], pitch_range[1]),
                cam_pivot,
                radius=cam_radius,
                device=device,
            )
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(z, conditioning_params, truncation_psi=args.trunc_psi, truncation_cutoff=args.trunc_cutoff)

            synth = G.synthesis(ws, camera_params, generate_background=True)  # TODO set to true...
            img = synth["image"]

            attr = torch.tensor(bisenet_attribute_chs["hair"]).to(device)
            parsing = mask_net(img)[0]
            parsing = parsing.squeeze(0).argmax(0)
            parsing = torch.where(parsing == attr.item(), torch.tensor(1).to(device), torch.tensor(0).to(device))
            hair_ratio = (torch.sum(parsing) / (parsing.shape[0] * parsing.shape[1])).item()
            # print(f"hair/image ratio for seed {seed}:", hair_ratio)

            # if (seed_idx % 2 == 0 and hair_ratio < hair_threshold_back) or (
            #     seed_idx % 2 == 1 and hair_ratio < hair_threshold_front
            # ):
            #     # print(f"Skipping image (seed {seed}) due to low hair ratio")
            #     continue

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            f_name = f"img_{created_samples}_seed0{seed:04d}"
            img_name = f"{f_name}.png"
            PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{args.output_dir}/{img_name}")

            camera_params_dict["labels"].append([img_name, camera_params.cpu().numpy().tolist()[0]])

            if args.save_original_w:
                camera_params_name = f"{f_name}.npy"
                np.save(f"{args.output_dir}/{camera_params_name}", ws.cpu().numpy()[0])

            created_samples += 1
            pbar.update(1)
        except Exception as e:
            print(f"Skipping image seed {seed} due to error: {e}")
            continue

    print("Done creating samples!")
    print(f"Created {created_samples} samples out of {args.num_samples} requested")

    with open(f"{args.output_dir}/dataset.json", "w") as f:
        json.dump(camera_params_dict, f)

    print("Saved dataset.json!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate samples from a trained PanoHead model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", required=True)
    parser.add_argument("--network_pkl", type=str, help="Path to the PanoHead model pickle file", required=True)
    parser.add_argument("--bisenet_ckpt", type=str, help="Path to the BiseNet model checkpoint file", required=True)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--trunc_psi", type=float, default=0.7, help="Truncation psi value")
    parser.add_argument("--trunc_cutoff", type=int, default=14, help="Truncation cutoff value")
    parser.add_argument("--fov_deg", type=float, default=18.837, help="Field of view in degrees")
    parser.add_argument("--pose_cond", type=int, default=90, help="Camera condition pose angle")
    parser.add_argument("--save_original_w", action="store_true", help="Save the original w vectors")
    parser.add_argument("--reload_generator", action="store_true", help="Reload the generator model")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Using args:", args)

    gen_samples(args)
