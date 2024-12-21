from PIL import Image, ImageOps
import torch
import numpy as np
import PIL.Image
import time
import traceback
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics


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

yaw_range_back = [(-np.pi / 2) + (np.pi / 3), (-np.pi / 2) - (np.pi / 3)]
yaw_range_front = [(np.pi / 2) - (np.pi / 3), (np.pi / 2) + (np.pi / 3)]
pitch_range = [np.pi / 2 - (np.pi / 5), np.pi / 2 + (np.pi / 5)]

# hair_threshold_back = 0.45  # gut feeling
# hair_threshold_front = 0.25  # gut feeling


def np_set_random_seed(num):
    seed = int(time.time()) % 1000000 + num
    np.random.seed(seed)
    return seed


def np_reset_seed():
    np.random.seed(2107)


def gen_synth_triplanes(G, mask_net=None, opts=None, idx=0, ready_camera_params=None, rendering_front_view=True):

    with torch.no_grad():
        try:
            device = opts.device

            cam_pivot = torch.tensor([0, 0, 0], device=device)
            cam_radius = G.rendering_kwargs.get("avg_camera_radius", 2.7)
            intrinsics = FOV_to_intrinsics(18.837, device=device)

            sample_fine = False

            while not sample_fine:
                np_set_random_seed(idx)

                z = torch.from_numpy(np.random.randn(opts.batch_size, 14, G.z_dim)).to(device)

                yaw_range = yaw_range_back if not rendering_front_view else yaw_range_front

                # front view conditioning
                conditioning_cam2world_pose = LookAtPoseSampler.sample(
                    np.random.uniform(yaw_range_front[0], yaw_range_front[1]),
                    np.random.uniform(pitch_range[0], pitch_range[1]),
                    cam_pivot,
                    radius=cam_radius,
                    device=device,
                )

                cam2world_pose = LookAtPoseSampler.sample(
                    np.random.uniform(yaw_range[0], yaw_range[1]),
                    np.random.uniform(pitch_range[0], pitch_range[1]),
                    cam_pivot,
                    radius=cam_radius,
                    device=device,
                )

                conditioning_camera_params = torch.cat(
                    [conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
                ).repeat(opts.batch_size, 1)

                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).repeat(
                    opts.batch_size, 1
                )

                ws = []

                for i in range(14):
                    w = G.mapping(
                        z[:, i, :],
                        conditioning_camera_params,
                        truncation_psi=opts.dynamic_synt_data_truncation_psi,
                        truncation_cutoff=opts.dynamic_synt_data_truncation_cutoff,
                    )

                    ws.append(w[:, 0, :])

                ws = torch.stack(ws, dim=1)

                synth = G.synthesis(
                    ws,
                    ready_camera_params if ready_camera_params is not None else camera_params,
                    generate_background=opts.dynamic_synth_data_generate_background,  # False
                    return_only_triplanes=False,
                )

                # if mask_net is not None:
                #     attr = torch.tensor(bisenet_attribute_chs["hair"]).to(device)
                #     parsing = mask_net(img)[0]
                #     parsing = parsing.squeeze(0).argmax(0)
                #     parsing = torch.where(parsing == attr.item(), torch.tensor(1).to(device), torch.tensor(0).to(device))
                #     hair_ratio = (torch.sum(parsing) / (parsing.shape[0] * parsing.shape[1])).item()

                #     if (not rendering_front_view and hair_ratio < hair_threshold_back) or (
                #         rendering_front_view and hair_ratio < hair_threshold_front
                #     ):
                #         continue

                # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                sample_fine = True

            np_reset_seed()

        except Exception as e:
            print(e)
            traceback.print_exc()
            return None

    return synth
