import os
import sys
import scipy
import imageio

os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from torch_utils import misc
import torch
import random
from dataset.dataset import ImageFolderDataset
import numpy as np
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch.utils.data import DataLoader, random_split
from PIL import Image
import argparse
import torchvision.transforms.functional as F
from torchvision import transforms
from training.volumetric_rendering.renderer import generate_planes, project_onto_planes
from shape_utils import convert_sdf_samples_to_ply

import mrcfile
import numpy as np

from training.triplanenet import TriPlaneNet
from tqdm import tqdm
import torch.nn.functional as F


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    return samples.unsqueeze(0), voxel_origin, voxel_size


face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def gen_interp_video(
    G,
    ws,
    triplane=None,
    mp4=None,
    seeds=None,
    shuffle_seed=None,
    w_frames=60 * 4,
    kind="cubic",
    grid_dims=(1, 1),
    num_keyframes=None,
    wraps=2,
    psi=1,
    truncation_cutoff=14,
    cfg="FFHQ",
    image_mode="image",
    gen_shapes=False,
    device=torch.device("cuda"),
    **video_kwargs,
):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    num_keyframes = 1
    camera_lookat_point = torch.tensor([0, 0, 0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(500, 1)

    # ws = ws.reshape(grid_h, grid_w, *ws.shape[1:]).unsqueeze(2).repeat(1, 1, num_keyframes, 1, 1)

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[0][yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    # video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    video_out = imageio.get_writer(mp4, mode="I", fps=60, codec="libx264", quality=9)
    outdir = "./"
    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.3
                yaw_range = 3.14 * 2
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14 / 2
                    + yaw_range
                    * frame_idx
                    / (num_keyframes * w_frames),  # np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    3.14 / 2 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    camera_lookat_point,
                    radius=2.7,
                    device=device,
                )
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                img = G.synthesis(
                    ws,
                    c[0:1],
                    ready_triplane=triplane,
                    generate_background=False,
                    make_background_white=True,
                    noise_mode="const",
                )["image"]

                # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                # img = (torch.cat(imgs, dim=0).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                # img = face_pool(img)  # TODO try avg pool
                imgs.append(img)

        video_out.append_data(layout_grid(torch.cat(imgs)))
    video_out.close()
    all_poses = np.stack(all_poses)


class Inference:
    def __init__(self, opts) -> None:
        self.opts = opts

        self.global_step = 0
        self.device = self.opts.device
        torch.backends.cudnn.benchmark = True

        SEED = 2107
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        self.net = TriPlaneNet(self.opts, decoder=None).to(self.device)
        self.net.requires_grad_(False)
        self.net.eval()

        self.visible_region_net = None

        if self.opts.visible_region_net_checkpoint_path is not None:
            self.opts.checkpoint_path = self.opts.visible_region_net_checkpoint_path

            self.visible_region_net = TriPlaneNet(self.opts, self.net.decoder, is_main_net=False).to(self.device)
            self.visible_region_net.requires_grad_(False)
            self.visible_region_net.eval()

        _, self.test_dataset = self.configure_datasets()
        self.configure_dataloaders(batch_size=1)

        self.gauss_kernel_0 = transforms.GaussianBlur(9, sigma=(1, 2))
        self.gauss_kernel_1 = transforms.GaussianBlur(9, sigma=(1, 2))
        self.gauss_kernel_2 = transforms.GaussianBlur(21, sigma=(4, 4))
        self.gauss_kernel_3 = transforms.GaussianBlur(51, sigma=(10, 10))

        self.erode = lambda tp, kernel_size: self.gauss_kernel_2(
            (
                1
                - F.max_pool2d(
                    (1 - tp),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                )
            )
        )

        self.dilate = lambda tp, kernel_size: self.gauss_kernel_2(
            F.max_pool2d(
                tp,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            )
        )

        self.yaw_range = None
        self.pitch_range = None
        self.intrinsics = None
        self.set_extreme_pose_list()
        self.plane_axes = generate_planes().to(self.device)

    def configure_datasets(self):

        full_dataset = ImageFolderDataset(
            path=self.opts.data_path,
            synth_path=self.opts.synth_data_path,
            load_original_ws=False,
            synth_data_ratio=1,
            use_labels=True,
        )

        train_size = 1
        test_size = len(full_dataset) - 1
        # train_size, test_size = (1.0 - 0.05), 0.05
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

    def set_extreme_pose_list(self):
        yaw_range_back = [(-np.pi / 2) + (np.pi / 2), (-np.pi / 2) - (np.pi / 2)]
        yaw_range_all = [0, np.pi * 2]

        self.intrinsics = FOV_to_intrinsics(self.opts.fov_deg, device=self.device)
        self.pitch_range = [np.pi / 2 - (np.pi / 5), np.pi / 2 + (np.pi / 5)]
        self.yaw_range = yaw_range_all if self.opts.fid_pose_cond == "all" else yaw_range_back

        print(f"Using yaw range {self.opts.fid_pose_cond}: {self.yaw_range}")

    def get_extreme_pose(self):
        cam_pivot = torch.tensor([0, 0, 0], device=self.device)
        cam_radius = self.net.decoder.rendering_kwargs.get("avg_camera_radius", 2.7)

        yaw = np.random.uniform(self.yaw_range[0], self.yaw_range[1])
        pitch = np.random.uniform(self.pitch_range[0], self.pitch_range[1])

        cam2world_pose = LookAtPoseSampler.sample(
            yaw,
            pitch,
            cam_pivot,
            radius=cam_radius,
            device=self.device,
        )

        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)

        return camera_params

    def infer(self):
        torch.cuda.empty_cache()

        for batch_idx, batch in tqdm(enumerate(self.test_dataloader), total=self.opts.num_samples):
            if batch_idx < self.opts.start_samples:
                continue

            if batch_idx - self.opts.start_samples >= self.opts.num_samples:
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

            x_resized, x, camera_param, x_mirror_resized, x_mirror, camera_param_mirror = (
                x_resized.to(self.device).float(),
                x.to(self.device).float(),
                camera_param.to(self.device).float(),
                x_mirror_resized.to(self.device).float(),
                x_mirror.to(self.device).float(),
                camera_param_mirror.to(self.device).float(),
            )

            if torch.any(w_original) and torch.any(synth_data):
                w_original = w_original.to(self.device).float()

            with torch.no_grad():
                if self.opts.only_use_first_stage:

                    codes = self.net.psp_encoder(x_resized)
                    codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)

                    outs = self.net.decoder.synthesis(
                        codes, camera_param, noise_mode="const", generate_background=self.opts.generate_background
                    )

                    outs["codes"] = codes
                    predicted_image = outs["image"]
                    predicted_image_to_save = (
                        (predicted_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    )

                elif self.opts.use_visible_region_stitching:
                    assert self.visible_region_net is not None, "Visible region net not loaded"

                    if self.opts.use_zplus or self.opts.use_z:
                        visible_outs = self.visible_region_net.get_initial_version(
                            x_resized,
                            camera_param,
                            camera_param_mirror,
                            generate_background=self.opts.generate_background,
                        )
                    else:
                        visible_outs = self.visible_region_net(
                            x_resized,
                            camera_param,
                            camera_param_mirror,
                            x_mirror=x_mirror_resized,
                            generate_background=self.opts.generate_background,
                        )
                    visible_codes = visible_outs["codes"]
                    visible_triplanes = visible_outs["triplanes"]

                    if self.opts.save_all_images_stiched:
                        visible_predicted_multi_image = self.plot_multi_view(
                            visible_codes, visible_triplanes, reverse=False
                        )

                    outs = self.net(
                        x_resized,
                        camera_param,
                        camera_param_mirror,
                        x_mirror=x_mirror_resized,
                        generate_background=self.opts.generate_background,
                    )

                    outs_codes = outs["codes"]
                    outs_triplanes = outs["triplanes"]

                    if self.opts.save_all_images_stiched:
                        outs_predicted_multi_image = self.plot_multi_view(outs_codes, outs_triplanes, reverse=False)

                    predicted_image = visible_outs["y_hat"]
                    predicted_image_to_save = (
                        (predicted_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    )

                    occluded_points_mask, visible_points_mask = self.get_occlusion_masks(visible_outs)

                    merged_triplanes = self.merge_triplanes(
                        visible_triplanes, outs_triplanes, visible_points_mask, occluded_points_mask
                    )

                    outs = {"codes": visible_codes, "triplanes": merged_triplanes}
                else:
                    outs = self.net(
                        x_resized,
                        camera_param,
                        camera_param_mirror,
                        x_mirror=x_mirror_resized,
                        generate_background=self.opts.generate_background,
                    )

                    predicted_image = outs["y_hat"]
                    predicted_image_to_save = (
                        (predicted_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    )

            codes = outs["codes"]
            triplanes = outs["triplanes"]

            if self.opts.generate_geometry:
                voxel_resolution = 512
                max_batch = 10000000

                samples, voxel_origin, voxel_size = create_samples(
                    N=voxel_resolution,
                    voxel_origin=[0, 0, 0],
                    cube_length=self.net.decoder.rendering_kwargs["box_warp"],
                )

                # print("sa", self.net.decoder.rendering_kwargs["box_warp"])

                samples = samples.to(self.device)
                sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=self.device)
                transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=self.device)
                transformed_ray_directions_expanded[..., -1] = -1

                head = 0
                with tqdm(total=samples.shape[1]) as pbar:
                    with torch.no_grad():
                        while head < samples.shape[1]:
                            torch.manual_seed(0)
                            sigma = self.net.decoder.sample_mixed_with_planes(
                                samples[:, head : head + max_batch],
                                transformed_ray_directions_expanded[:, : samples.shape[1] - head],
                                triplanes,
                            )["sigma"]

                            sigmas[:, head : head + max_batch] = sigma
                            head += max_batch
                            pbar.update(max_batch)

                sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                sigmas = np.flip(sigmas, 0)

                pad = int(30 * voxel_resolution / 256)
                pad_top = int(38 * voxel_resolution / 256)
                sigmas[:pad] = 0
                sigmas[-pad:] = 0
                sigmas[:, :pad] = 0
                sigmas[:, -pad_top:] = 0
                sigmas[:, :, :pad] = 0
                sigmas[:, :, -pad:] = 0

                geo_path = os.path.join(self.opts.output_dir, "geometry", f"{fname[0]}_shape")

                if self.opts.geometry_output_ply:
                    geo_path += ".ply"

                    convert_sdf_samples_to_ply(
                        np.transpose(sigmas, (2, 1, 0)),
                        [0, 0, 0],
                        1,
                        geo_path,
                        level=10,
                    )
                else:  # output mrc
                    geo_path += ".mrc"

                    with mrcfile.new_mmap(geo_path, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                        mrc.data[:] = sigmas

            if self.opts.save_videos:
                gen_interp_video(
                    self.net.decoder,
                    codes,
                    triplane=triplanes,
                    mp4=f"{self.opts.output_dir}/videos/{fname[0]}.mp4",
                    image_mode="image",
                    device=self.device,
                    w_frames=240,
                )

                Image.open(os.path.join(self.opts.synth_data_path, fname[0])).resize((256, 256)).save(
                    f"{self.opts.output_dir}/videos/{fname[0]}"
                )

            if self.opts.save_images:
                if self.opts.use_visible_region_stitching and self.opts.save_all_images_stiched:
                    predicted_multi_image_to_save = self.plot_multi_view(codes, triplanes, reverse=False)
                    self.save_all_images_stiched(
                        predicted_image_to_save[0].cpu().numpy(),
                        visible_predicted_multi_image,
                        outs_predicted_multi_image,
                        predicted_multi_image_to_save,
                        fname[0],
                    )
                else:
                    predicted_image_to_save = predicted_image_to_save[0].cpu().numpy()
                    predicted_multi_image_to_save = self.plot_multi_view(codes, triplanes, reverse=False)
                    self.save_all_images(
                        predicted_image_to_save, predicted_multi_image_to_save, fname[0], synth_data[0]
                    )

    def plot_multi_view(self, ws, triplane, visible_triplanes=None, visible_sample_coordinates=None, reverse=False):
        angle_p = -0.2
        cam_pivot = torch.tensor([0, 0, 0], device=self.device)
        cam_radius = self.net.decoder.rendering_kwargs.get("avg_camera_radius", 2.7)
        intrinsics = FOV_to_intrinsics(self.opts.fov_deg, device=self.device)

        imgs = []

        for _, angles in enumerate(
            [
                # (np.pi, angle_p),
                # (135 / 180 * np.pi, angle_p),
                (90 / 180 * np.pi, angle_p),
                (45 / 180 * np.pi, angle_p),
                (0, angle_p),
                (-45 / 180 * np.pi, angle_p),
                (-90 / 180 * np.pi, angle_p),
                (-135 / 180 * np.pi, angle_p),
                (-np.pi, angle_p),
            ]
        ):
            angle_y = angles[0] * (-1 if reverse else 1)
            angle_p = angles[1]
            # rand camera setting
            cam2world_pose = LookAtPoseSampler.sample(
                np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=self.device
            )
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            synth = self.net.decoder.synthesis(
                ws,
                camera_params,
                ready_triplane=triplane,
                noise_mode="const",
                ready_triplane_visible=visible_triplanes,
                visible_sample_coordinates=visible_sample_coordinates,
                generate_background=self.opts.generate_background,
                make_background_white=self.opts.make_background_white,
            )
            img = synth["image"]

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)

        img = torch.cat(imgs, dim=2)

        return img[0].cpu().numpy()

    def save_all_images_stiched(self, predicted_image, visible_images, occluded_images, resulting_images, fname):
        images = [
            Image.open(os.path.join(self.opts.synth_data_path, fname)),
            Image.fromarray(predicted_image),
            Image.fromarray(visible_images),
            Image.fromarray(occluded_images),
            Image.fromarray(resulting_images),
        ]

        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = sum(heights)

        new_im = Image.new("RGB", (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        path = os.path.join(self.opts.output_dir, fname)
        new_im.save(path)

    def save_all_images(self, predicted_image, predicted_multi_image, fname, synth_data):

        images = [
            Image.open(os.path.join(self.opts.data_path if not synth_data else self.opts.synth_data_path, fname)),
            Image.fromarray(predicted_image),
            Image.fromarray(predicted_multi_image),
        ]

        min_height = min([image.size[1] for image in images])

        new_images = []

        for image in images:
            if image.size[0] != min_height and image.size[0] == image.size[1]:
                image = image.resize((min_height, min_height))

            new_images.append(image)

        images = new_images

        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        path = os.path.join(self.opts.output_dir, fname)
        new_im.save(path)

    def get_occlusion_masks(self, rec_img_dict):

        bsz = rec_img_dict["triplanes"].shape[0]

        with torch.no_grad():
            xyz = rec_img_dict["xyz_final"]  # surface xyz       B 64x64  3

            x = (xyz[:, :, 0] + 0.5) * 256
            y = (xyz[:, :, 1] + 0.5) * 256
            z = (xyz[:, :, 2] + 0.5) * 256
            x = torch.clamp(x, min=0, max=255)
            y = torch.clamp(y, min=0, max=255)
            z = torch.clamp(z, min=0, max=255)
            x = x.long()
            y = y.long()
            z = z.long()

            visible_points_mask = torch.zeros((bsz, 3, 256, 256))
            for j in range(bsz):
                for k in range(3):
                    if k == 0:
                        visible_points_mask[j, k, z[j], y[j]] = 1
                    elif k == 1:
                        visible_points_mask[j, k, z[j], x[j]] = 1
                    elif k == 2:
                        visible_points_mask[j, k, z[j], y[j]] = 1

                    visible_points_mask[j, k] = self.gauss_kernel_0(visible_points_mask[j, k : k + 1])
                    visible_points_mask[j, k] = torch.where(visible_points_mask[j, k : k + 1] > 0.01, 1, 0).float()
                    visible_points_mask[j, k] = self.gauss_kernel_1(visible_points_mask[j, k : k + 1])

                    if k == 2:
                        h = 256
                        visible_points_mask[j, k] = self.dilate(visible_points_mask[j, k : k + 1], 21)
                        visible_points_mask[j, k, : h // 2, :] = 0

                    visible_points_mask[j, k] = self.dilate(visible_points_mask[j, k : k + 1], 11)
                    visible_points_mask[j, k] = self.gauss_kernel_3(visible_points_mask[j, k : k + 1])

            visible_points_mask = visible_points_mask.to(self.device)
            occluded_points_mask = 1 - visible_points_mask

            return occluded_points_mask, visible_points_mask

    def merge_triplanes(self, visible_triplane, occluded_triple, visible_points_mask, occluded_points_mask):
        triplane = torch.zeros_like(visible_triplane)

        triplane[:, 0, :, :, :] = (visible_points_mask[:, 0, :, :]).unsqueeze(1) * visible_triplane[
            :, 0, :, :, :
        ] + occluded_points_mask[:, 0, :, :].unsqueeze(1) * visible_triplane[:, 0, :, :, :]

        triplane[:, 1, :, :, :] = (visible_points_mask[:, 1, :, :]).unsqueeze(1) * visible_triplane[
            :, 1, :, :, :
        ] + occluded_points_mask[:, 1, :, :].unsqueeze(1) * occluded_triple[:, 1, :, :, :]

        triplane[:, 2, :, :, :] = (visible_points_mask[:, 2, :, :]).unsqueeze(1) * visible_triplane[
            :, 2, :, :, :
        ] + occluded_points_mask[:, 2, :, :].unsqueeze(1) * occluded_triple[:, 2, :, :, :]

        return triplane


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained PanoHead model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", required=True)
    parser.add_argument("--checkpoint_path", type=str, help="Path to the PanoHead encoder checkpoint", required=True)
    parser.add_argument("--data_path", type=str, help="Path to data images.", required=True)
    parser.add_argument("--synth_data_path", type=str, help="Path to syhtn data images.", required=True)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--trunc_psi", type=float, default=0.7, help="Truncation psi value")
    parser.add_argument("--trunc_cutoff", type=int, default=14, help="Truncation cutoff value")
    parser.add_argument("--fov_deg", type=float, default=18.837, help="Field of view in degrees")
    parser.add_argument("--pose_cond", type=int, default=90, help="Camera condition pose angle")
    parser.add_argument("--n_styles", type=int, default=14, help="Number of styles for encoder")
    parser.add_argument("--only_use_first_stage", type=int, default=0, help="Use only the first stage of the model")
    parser.add_argument("--use_pixelshuffle", type=int, default=0, help="Use pixel shuffle for the decoder")
    parser.add_argument("--generate_background", type=int, default=0, help="Generate background for the decoder")

    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--synth_data_ratio", type=float, default=0, help="Ratio of synthetic data to use")
    parser.add_argument("--make_background_white", type=int, default=1, help="Use Z for initial styles")
    parser.add_argument("--save_images", type=int, default=1, help="Save images")
    parser.add_argument("--save_all_images_stiched", type=int, default=0, help="Save all images stiched")

    parser.add_argument(
        "--visible_region_net_checkpoint_path", type=str, default=None, help="Path to the visible region net checkpoint"
    )
    parser.add_argument("--use_visible_region_stitching", type=int, default=0, help="Use visible region stitching")

    parser.add_argument("--generate_geometry", type=int, default=0, help="Generate geometry")
    parser.add_argument("--geometry_output_ply", type=int, default=1, help="Output geometry as ply")
    parser.add_argument("--render_geometry", type=int, default=1, help="Render geometry")
    parser.add_argument("--save_videos", type=int, default=1, help="Save videos")

    opts = parser.parse_args()
    print(opts)

    opts.output_dir += f"/{opts.checkpoint_path.split('/')[-1]}/"

    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, "geometry"), exist_ok=True)
    os.makedirs(os.path.join(opts.output_dir, "videos"), exist_ok=True)

    inference = Inference(opts)

    inference.infer()
