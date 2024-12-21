import copy

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import legacy

import dnnlib
from training.encoder import encoders as psp_encoders
from training.triplane import TriPlaneGenerator
from training.encoder.triplane_encoders import TriPlaneEncoder
from configs.paths_config import model_paths
from torch_utils import misc


class TriPlaneNet(nn.Module):
    def __init__(self, opts, decoder=None):
        super(TriPlaneNet, self).__init__()
        self.opts = opts

        self.decoder = decoder
        self.set_panohead_generator()

        self.set_latent_avg()
        self.set_psp_encoder()
        self.set_triplane_encoder()
        self.load_weights()

    def forward(
        self,
        x,
        camera_params,
        mirror_camera_params,
        novel_view_camera_params=None,
        x_mirror=None,
        use_background_encoder=False,
        generate_background=True,
        make_background_white=False,
    ):
        x_clone = x.clone().detach()

        if novel_view_camera_params is None:
            novel_view_camera_params = mirror_camera_params

        initial_version = self.get_initial_version(
            x, camera_params.clone().detach(), mirror_camera_params, use_background_encoder, generate_background
        )

        new_codes = initial_version["codes"].clone().detach()
        y_hat_initial_clone = initial_version["y_hat_resized"].clone().detach()
        y_hat_initial_mirror_clone = initial_version["y_hat_novel_resized"].clone().detach()

        x_input = torch.cat(
            [y_hat_initial_clone, x_clone - y_hat_initial_clone, x_mirror - y_hat_initial_mirror_clone], dim=1
        )

        ready_triplane = self.triplanenet_encoder(x_input, initial_version["triplanes"].clone().detach())

        outs = self.decoder.synthesis(
            new_codes,
            camera_params.clone().detach(),
            ws_bcg=None,
            ready_triplane=ready_triplane,
            noise_mode="const",
            generate_background=generate_background,
            make_background_white=make_background_white,
        )

        y_hat = outs["image"]
        y_hat_resized = F.adaptive_avg_pool2d(y_hat, (256, 256))

        outs["y_hat"] = y_hat
        outs["y_hat_resized"] = y_hat_resized
        outs["codes"] = new_codes
        outs["initial_version"] = initial_version

        if novel_view_camera_params is not None:
            outs_novel = self.decoder.synthesis(
                new_codes,
                novel_view_camera_params.clone().detach(),
                ws_bcg=None,
                ready_triplane=ready_triplane,
                noise_mode="const",
                generate_background=generate_background,
            )
            y_hat_novel = outs_novel["image"]
            y_hat_novel_resized = F.adaptive_avg_pool2d(y_hat_novel, (256, 256))
            outs["y_hat_novel"] = y_hat_novel
            outs["y_hat_novel_resized"] = y_hat_novel_resized
            outs["depth_novel"] = outs_novel["image_depth"]

        return outs

    def get_initial_version(
        self,
        x,
        camera_params,
        novel_view_camera_params=None,
        use_background_encoder=False,
        generate_background=True,
    ):

        codes = self.psp_encoder(x)
        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        background_codes = None

        outs = self.decoder.synthesis(
            codes, camera_params, ws_bcg=background_codes, noise_mode="const", generate_background=generate_background
        )

        y_hat = outs["image"]
        y_hat_resized = F.adaptive_avg_pool2d(y_hat, (256, 256))

        outputs = {
            "xyz_final": outs["xyz_final"],
            "codes": codes,
            "background_codes": background_codes if use_background_encoder else None,
            "y_hat": y_hat,
            "y_hat_resized": y_hat_resized,
            "triplanes": outs["triplanes"],
            "image_depth": outs["image_depth"],
            "mask": outs["image_mask"],
            "ray_origins": outs["ray_origins"],
            "ray_directions": outs["ray_directions"],
        }

        if novel_view_camera_params is not None:
            y_hat_novel = self.decoder.synthesis(
                codes,
                novel_view_camera_params,
                ws_bcg=background_codes,
                noise_mode="const",
                generate_background=generate_background,
            )["image"]

            y_hat_novel_resized = F.adaptive_avg_pool2d(y_hat_novel, (256, 256))
            outputs["y_hat_novel"] = y_hat_novel
            outputs["y_hat_novel_resized"] = y_hat_novel_resized
            return outputs

        return outputs

    def set_psp_encoder(self):
        self.psp_encoder = psp_encoders.GradualStyleEncoder(50, "ir_se", self.opts)

    def set_triplane_encoder(self):
        self.triplanenet_encoder = TriPlaneEncoder(self.opts)

    def set_latent_avg(self):
        self.latent_avg = torch.from_numpy(np.load(model_paths["latent_avg"])).to(self.opts.device)

    def set_panohead_generator(self):
        if self.decoder is None:
            with dnnlib.util.open_url(model_paths["pano_head"]) as f:
                G = legacy.load_network_pkl(f)["G_ema"].to(self.opts.device)  # type: ignore

            G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(self.opts.device)
            misc.copy_params_and_buffers(G, G_new, require_all=True)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G_new.rendering_kwargs = G.rendering_kwargs
            self.decoder = G_new

    def load_weights(self):
        if self.opts.checkpoint_path is None:
            print("Loading encoders weights from irse50!")
            encoder_ckpt = torch.load(model_paths["ir_se50"])
            self.psp_encoder.load_state_dict(encoder_ckpt, strict=False)

            shape = encoder_ckpt["input_layer.0.weight"].shape
            altered_input_layer = torch.randn(shape[0], 9, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = encoder_ckpt["input_layer.0.weight"]
            encoder_ckpt["input_layer.0.weight"] = altered_input_layer
            self.triplanenet_encoder.load_state_dict(encoder_ckpt, strict=False)
        else:
            checkpoint = torch.load(self.opts.checkpoint_path, map_location="cpu")["state_dict"]
            self.load_state_dict(checkpoint, strict=False)
            print(f"Loading encoders weights from {self.opts.checkpoint_path}")
