import os
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss
from criteria.lpips.lpips import LPIPS
from training.triplanenet import TriPlaneNet
from training.ranger import Ranger
from training.encoder.discriminator import TriplaneDiscriminator2D
from dataset.dataset import ImageFolderDataset
from dataset.synth_random_triplane_data import gen_synth_triplanes
from configs.paths_config import dataset_paths, model_paths
import random
from torchvision import transforms
from face_parsing_pytorch.model import BiSeNet
import numpy as np
import json

matplotlib.use("Agg")

EPS = 1


def get_panohead_camera_params():
    with open(os.path.join(dataset_paths["train"], "dataset.json"), "r") as f:
        panohead_cam_list = json.load(f)["labels"]

    panohead_cam_list = [[x[1]] for x in panohead_cam_list]
    return panohead_cam_list


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0
        self.device = self.opts.device
        torch.backends.cudnn.benchmark = True
        SEED = 2107
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        self.gauss_kernel_0 = transforms.GaussianBlur(9, sigma=(1, 2))
        self.gauss_kernel_1 = transforms.GaussianBlur(9, sigma=(1, 2))
        self.erode = (
            lambda input_tensor, kernel_size: torch.nn.functional.conv2d(
                input_tensor.unsqueeze(0).float(),
                torch.ones(1, 1, kernel_size, kernel_size, device=input_tensor.device).float(),
                padding=kernel_size // 2,
            )
            .clamp(0, 1)
            .byte()
            .squeeze(0)
        )

        self.gauss_kernel_3 = transforms.GaussianBlur(51, sigma=(10, 10))

        self.dilate = lambda tp, kernel_size: self.gauss_kernel_2(
            F.max_pool2d(
                tp,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            )
        )

        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger

            self.wb_logger = WBLogger(self.opts)

        # self.mask_net = BiSeNet(n_classes=19).to(self.device).requires_grad_(False)
        # self.mask_net.load_state_dict(torch.load(model_paths["bisenet"], map_location=self.device))
        # self.mask_net.eval()

        # Initialize network
        self.net = TriPlaneNet(self.opts).to(self.device)
        if self.opts.use_discriminator:
            self.discriminator = TriplaneDiscriminator2D(self.opts).to(self.device)

        self.panohead_cam_list = get_panohead_camera_params()

        self.mse_loss = nn.MSELoss().to(self.device).eval()

        if self.opts.lpips_lambda_psp > 0 or self.opts.lpips_lambda_triplane > 0:
            self.lpips_loss = LPIPS(net_type="alex").to(self.device).eval()

        if self.opts.id_lambda_psp > 0 or self.opts.id_lambda_triplane > 0 or self.opts.id_lambda_triplane_mirror > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()

        if self.opts.w_lambda_psp > 0:
            self.w_loss = nn.MSELoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer_psp = self.configure_psp_optimizers()
        self.optimizer_triplane = self.configure_triplane_optimizers()
        if self.opts.use_discriminator:
            self.optimizer_discriminator = self.configure_discriminator_optimizers()

        self.net.psp_encoder.requires_grad_(False)
        self.net.triplanenet_encoder.requires_grad_(False)
        if self.opts.use_discriminator:
            self.discriminator.requires_grad_(False)

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.configure_dataloaders(self.opts.batch_size)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None

        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        # Resume training process from checkpoint path
        if self.opts.checkpoint_path is not None:
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")

            if not self.opts.use_Z:
                if "step" in ckpt:
                    self.global_step = ckpt["step"]
                    print(f"Resuming training process from step {self.global_step}")

                if "psp_optimizer" in ckpt:
                    self.optimizer_psp.load_state_dict(ckpt["psp_optimizer"])
                    print("Load psp optimizer from checkpoint")

                if "triplane_optimizer" in ckpt:
                    self.optimizer_triplane.load_state_dict(ckpt["triplane_optimizer"])
                    print("Load triplane optimizer from checkpoint")

                if "discriminator_optimizer" in ckpt:
                    self.optimizer_discriminator.load_state_dict(ckpt["discriminator_optimizer"])
                    print("Load discriminator optimizer from checkpoint")

                if "best_val_loss" in ckpt:
                    self.best_val_loss = ckpt["best_val_loss"]
                    print(f"Current best val loss: {self.best_val_loss}")

    def train(self):
        self.net.train()
        torch.cuda.empty_cache()

        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):

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

                if torch.any(w_original) and torch.any(synth_data) and self.opts.w_lambda_psp > 0:
                    w_original = w_original.to(self.device).float()

                loss_dict = {}
                loss = None
                loss_psp = None

                if self.global_step < self.opts.stop_psp:
                    self.net.psp_encoder.requires_grad_(True)
                    if self.opts.mirror_lambda > 0:
                        initial_outs = self.net.get_initial_version(
                            x_resized.clone().detach(),
                            camera_param.clone().detach(),
                            camera_param_mirror.clone().detach(),
                            generate_background=self.opts.generate_background,
                        )

                        loss_psp, loss_dict, id_logs = self.calc_loss_psp(
                            x_resized.clone().detach(),
                            initial_outs["y_hat_resized"],
                            initial_outs["codes"],
                            w_original,
                            synth_data,
                            loss_dict,
                        )

                        loss_psp_mirror, loss_dict, _ = self.calc_mirror_loss_psp(
                            x_resized.clone().detach(),
                            x_mirror_resized.clone().detach(),
                            initial_outs["y_hat_novel_resized"],
                            loss_dict,
                        )

                        loss_psp += self.opts.mirror_lambda * loss_psp_mirror
                    else:
                        initial_outs = self.net.get_initial_version(
                            x_resized.clone().detach(),
                            camera_param.clone().detach(),
                            None,
                            generate_background=self.opts.generate_background,
                        )

                        loss_psp, loss_dict, id_logs = self.calc_loss_psp(
                            x_resized.clone().detach(),
                            initial_outs["y_hat_resized"],
                            initial_outs["codes"],
                            w_original,
                            synth_data,
                            loss_dict,
                        )

                    if loss_psp is not None:
                        if torch.isnan(loss_psp).any():
                            self.optimizer_psp.zero_grad()
                        else:
                            self.optimizer_psp.zero_grad()
                            loss_psp.backward()
                            self.optimizer_psp.step()
                            self.optimizer_psp.zero_grad()

                    if (
                        self.opts.use_discriminator
                        and self.global_step >= self.opts.add_discriminator
                        and self.global_step < self.opts.stop_psp
                        and self.opts.apply_d_loss_on_psp
                    ):

                        initial_outs_dicts = self.net.get_initial_version(
                            x_resized.clone().detach(),
                            camera_param.clone().detach(),
                            camera_param_mirror.clone().detach(),
                            generate_background=self.opts.generate_background,
                        )

                        occluded_triplanes, visible_mask = self.get_occluded_triplanes(initial_outs_dicts)

                        if visible_mask is not None:
                            occlusion_mask = 1 - visible_mask
                            resized_occlusion_mask = F.interpolate(
                                occlusion_mask, size=(32, 32), mode="bilinear", align_corners=False
                            )

                        fake_preds = self.discriminator(occluded_triplanes)
                        if self.opts.discriminator_name == "TriplaneDiscriminator2DLossMasking":
                            loss_G_adv = self.g_nonsaturating_loss(fake_preds * resized_occlusion_mask)
                        else:
                            loss_G_adv = self.g_nonsaturating_loss(fake_preds)
                        loss_dict["loss_G_adv_psp"] = float(loss_G_adv)
                        loss_G_adv = loss_G_adv * self.opts.adv_lambda

                        if torch.isnan(loss_G_adv).any():
                            self.optimizer_psp.zero_grad()
                        else:
                            self.optimizer_psp.zero_grad()
                            loss_G_adv.backward()
                            self.optimizer_psp.step()
                            self.optimizer_psp.zero_grad()

                    self.net.psp_encoder.requires_grad_(False)

                outs = None

                if self.global_step >= self.opts.add_triplanenet:
                    self.net.triplanenet_encoder.requires_grad_(True)

                    if self.opts.mirror_lambda > 0:
                        outs = self.net.forward(
                            x_resized.clone().detach(),
                            camera_param.clone().detach(),
                            camera_param_mirror.clone().detach(),
                            x_mirror=x_mirror_resized,
                            generate_background=self.opts.generate_background,
                        )
                        loss, loss_dict, id_logs = self.calc_loss_triplane(
                            x_resized.clone().detach(), outs["y_hat_resized"], loss_dict
                        )

                        loss_mirror, loss_dict, _ = self.calc_mirror_loss_triplane(
                            x_resized.clone().detach(),
                            x_mirror_resized.clone().detach(),
                            outs["y_hat_novel_resized"],
                            loss_dict,
                        )

                        loss += self.opts.mirror_lambda * loss_mirror
                    else:
                        outs = self.net.forward(
                            x_resized.clone().detach(),
                            camera_param.clone().detach(),
                            camera_param_mirror.clone().detach(),
                            x_mirror=x_mirror_resized,
                            generate_background=self.opts.generate_background,
                        )
                        loss, loss_dict, id_logs = self.calc_loss_triplane(
                            x_resized.clone().detach(), outs["y_hat_resized"], loss_dict
                        )

                    if loss is not None:
                        if torch.isnan(loss).any():
                            self.optimizer_triplane.zero_grad()
                        else:
                            self.optimizer_triplane.zero_grad()
                            loss.backward()
                            self.optimizer_triplane.step()
                            self.optimizer_triplane.zero_grad()

                    if (
                        self.opts.use_discriminator
                        and self.global_step >= self.opts.add_discriminator
                        and self.global_step >= self.opts.add_triplanenet
                        and self.opts.apply_d_loss_on_triplanenet
                    ):

                        outs_disc = self.net.forward(
                            x_resized.clone().detach(),
                            camera_param.clone().detach(),
                            camera_param_mirror.clone().detach(),
                            x_mirror=x_mirror_resized,
                            generate_background=self.opts.generate_background,
                        )

                        occluded_triplanes, visible_mask = self.get_occluded_triplanes(outs_disc)

                        if visible_mask is not None:
                            occlusion_mask = 1 - visible_mask
                            resized_occlusion_mask = F.interpolate(
                                occlusion_mask, size=(32, 32), mode="bilinear", align_corners=False
                            )

                        fake_preds = self.discriminator(occluded_triplanes)

                        if self.opts.discriminator_name == "TriplaneDiscriminator2DLossMasking":
                            loss_G_adv = self.g_nonsaturating_loss(fake_preds * resized_occlusion_mask)
                        else:
                            loss_G_adv = self.g_nonsaturating_loss(fake_preds)

                        loss_dict["loss_G_adv_triplanenet"] = float(loss_G_adv)
                        loss_G_adv = loss_G_adv * self.opts.adv_lambda

                        if loss_G_adv is not None:
                            if torch.isnan(loss_G_adv).any():
                                self.optimizer_triplane.zero_grad()
                            else:
                                self.optimizer_triplane.zero_grad()
                                loss_G_adv.backward()
                                self.optimizer_triplane.step()
                                self.optimizer_triplane.zero_grad()

                    self.net.triplanenet_encoder.requires_grad_(False)

                if self.opts.use_discriminator and self.global_step >= self.opts.add_discriminator:
                    self.discriminator.requires_grad_(True)

                    real_dict = gen_synth_triplanes(
                        self.net.decoder, mask_net=None, opts=self.opts, idx=self.global_step
                    )
                    generated_dict = (
                        initial_outs if self.global_step < self.opts.stop_psp and initial_outs is not None else outs
                    )

                    real_occluded_triplanes, real_visible_mask = self.get_occluded_triplanes(real_dict)
                    generated_occluded_triplanes, generated_visible_mask = self.get_occluded_triplanes(generated_dict)

                    generated_occlusion_mask = None
                    real_occlusion_mask = None

                    if real_visible_mask is not None and generated_visible_mask is not None:
                        real_occlusion_mask = 1 - real_visible_mask
                        generated_occlusion_mask = 1 - generated_visible_mask

                    d_loss, loss_dict = self.calc_discriminator_loss(
                        loss_dict,
                        generated_occluded_triplanes.clone().detach(),
                        real_occluded_triplanes.clone().detach(),
                        generated_occlusion_mask.clone().detach() if generated_occlusion_mask is not None else None,
                        real_occlusion_mask.clone().detach() if real_occlusion_mask is not None else None,
                    )

                    if d_loss is not None:
                        if torch.isnan(d_loss).any():
                            self.optimizer_discriminator.zero_grad()
                        else:
                            self.optimizer_discriminator.zero_grad()
                            d_loss.backward()
                            self.optimizer_discriminator.step()

                    # R1 Regularization
                    if self.global_step % self.opts.d_reg_every == 0:
                        d_r1_loss, loss_dict = self.calc_discriminator_r1_loss(
                            loss_dict, real_occluded_triplanes.clone().detach()
                        )

                        if d_r1_loss is not None:
                            if torch.isnan(d_r1_loss).any():
                                self.optimizer_discriminator.zero_grad()
                            else:
                                self.optimizer_discriminator.zero_grad()
                                d_r1_loss.backward()
                                self.optimizer_discriminator.step()

                    self.discriminator.requires_grad_(False)
                    self.optimizer_discriminator.zero_grad()

                if (
                    (self.global_step < self.opts.stop_psp and loss_psp is not None and torch.isnan(loss_psp).any())
                    or (self.global_step >= self.opts.add_triplanenet and loss is not None and torch.isnan(loss).any())
                    or (
                        self.global_step >= self.opts.add_discriminator
                        and self.opts.use_discriminator
                        and d_loss is not None
                        and torch.isnan(d_loss).any()
                    )
                ):  # discriminator loss
                    print("NaN detected, skipping step")
                    continue

                if outs is None:
                    final_out = initial_outs["y_hat_resized"]
                    if "y_hat_novel_resized" in initial_outs:
                        final_out_novel = initial_outs["y_hat_novel_resized"]
                    else:
                        final_out_novel = final_out

                else:
                    final_out = outs["y_hat"]
                    if "y_hat_novel" in outs:
                        final_out_novel = outs["y_hat_novel"]
                    else:
                        final_out_novel = final_out

                if self.global_step % self.opts.image_interval == 0 or (
                    self.global_step < 1000 and self.global_step % 25 == 0
                ):
                    # id_logs is defined
                    self.parse_and_log_images(id_logs, x, final_out, final_out_novel, title="images/train")

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix="train")
                    self.log_metrics(loss_dict, prefix="train")

                # Log images of first batch to wandb
                if self.opts.use_wandb and batch_idx == 0:  # and self.phase == "first"
                    # TODO implement triplanenet second stage encoder and change the names below
                    self.wb_logger.log_images_to_wandb(
                        x, final_out, final_out_novel, id_logs, prefix="train", step=self.global_step, opts=self.opts
                    )

                # Validation related
                val_loss_dict = None
                if (
                    self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps
                ) and not self.opts.debug:
                    val_loss_dict = self.validate()

                    if val_loss_dict is not None:
                        if "loss_triplane" in val_loss_dict:
                            new_val_loss = val_loss_dict["loss_triplane"]
                        else:
                            new_val_loss = val_loss_dict["loss_psp"]

                    if val_loss_dict and (self.best_val_loss is None or new_val_loss < self.best_val_loss):
                        self.best_val_loss = new_val_loss
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print("OMG, finished training!")
                    break

                self.global_step += 1

    def validate(self):
        torch.cuda.empty_cache()
        self.net.eval()
        agg_loss_dict = []

        for batch_idx, batch in enumerate(self.test_dataloader):
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
                cur_loss_dict = {}
                outs = None
                sampled_camera_params = self.randomly_sample_camera_poses(x.size(0))

                if self.global_step < self.opts.stop_psp:
                    if self.opts.mirror_lambda > 0:
                        initial_outs = self.net.get_initial_version(
                            x_resized,
                            camera_param,
                            camera_param_mirror,
                        )
                        loss_psp, cur_loss_dict, id_logs = self.calc_loss_psp(
                            x_resized,
                            initial_outs["y_hat_resized"],
                            initial_outs["codes"],
                            w_original,
                            synth_data,
                            cur_loss_dict,
                        )
                        loss_psp_mirror, cur_loss_dict, _ = self.calc_mirror_loss_psp(
                            x_resized,
                            x_mirror_resized,
                            initial_outs["y_hat_novel_resized"],
                            cur_loss_dict,
                        )
                        initial_outs = self.net.get_initial_version(x_resized, camera_param, sampled_camera_params)
                    else:
                        initial_outs = self.net.get_initial_version(x_resized, camera_param, None)

                        loss_psp, cur_loss_dict, id_logs = self.calc_loss_psp(
                            x_resized.clone().detach(),
                            initial_outs["y_hat_resized"],
                            initial_outs["codes"],
                            w_original,
                            synth_data,
                            cur_loss_dict,
                        )
                        initial_outs = self.net.get_initial_version(x_resized, camera_param, sampled_camera_params)

                if self.global_step >= self.opts.add_triplanenet:
                    if self.opts.mirror_lambda > 0:
                        outs = self.net.forward(
                            x_resized,
                            camera_param,
                            camera_param_mirror,
                            x_mirror=x_mirror_resized,
                        )
                        loss, cur_loss_dict, id_logs = self.calc_loss_triplane(
                            x_resized, outs["y_hat_resized"], cur_loss_dict
                        )
                        loss_mirror, cur_loss_dict, _ = self.calc_mirror_loss_triplane(
                            x_resized,
                            x_mirror_resized,
                            outs["y_hat_novel_resized"],
                            cur_loss_dict,
                        )

                        outs = self.net.forward(
                            x_resized, camera_param, sampled_camera_params, x_mirror=x_mirror_resized
                        )
                    else:
                        outs = self.net.forward(
                            x_resized,
                            camera_param,
                            camera_param_mirror,
                            x_mirror=x_mirror_resized,
                        )
                        loss, cur_loss_dict, id_logs = self.calc_loss_triplane(
                            x_resized, outs["y_hat_resized"], cur_loss_dict
                        )
                        outs = self.net.forward(
                            x_resized, camera_param, sampled_camera_params, x_mirror=x_mirror_resized
                        )

            agg_loss_dict.append(cur_loss_dict)

            if outs is None:
                final_out = initial_outs["y_hat_resized"]
                if "y_hat_novel_resized" in initial_outs:
                    final_out_novel = initial_outs["y_hat_novel_resized"]
                else:
                    final_out_novel = final_out
            else:
                final_out = outs["y_hat"]
                if "y_hat_novel" in outs:
                    final_out_novel = outs["y_hat_novel"]
                else:
                    final_out_novel = final_out

            self.parse_and_log_images(
                id_logs, x, final_out, final_out_novel, title="images/test", subscript="{:04d}".format(batch_idx)
            )

            # Log images of first batch to wandb
            if self.opts.use_wandb and batch_idx == 0:
                self.wb_logger.log_images_to_wandb(
                    x, final_out, final_out_novel, id_logs, prefix="test", step=self.global_step, opts=self.opts
                )

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)

        self.log_metrics(loss_dict, prefix="test")
        self.print_metrics(loss_dict, prefix="test")

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = "best_model.pt" if is_best else f"iteration_{self.global_step}.pt"
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, "timestamp.txt"), "a") as f:
            if is_best:
                f.write(f"**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n")
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f"Step - {self.global_step}, \n{loss_dict}\n")

    def configure_psp_optimizers(self):
        params = list(self.net.psp_encoder.parameters())
        if self.opts.optim_name_psp == "adam":
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_psp)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate_psp)
        return optimizer

    def configure_triplane_optimizers(self):
        params = list(self.net.triplanenet_encoder.parameters())
        if self.opts.optim_name_triplane == "adam":
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_triplane)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate_triplane)
        return optimizer

    def configure_discriminator_optimizers(self):
        # TODO try with Ranger ==>
        params = list(self.discriminator.parameters())
        optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_dis)
        return optimizer

    def configure_datasets(self):

        full_dataset = ImageFolderDataset(
            path=dataset_paths["train"],
            synth_path=dataset_paths["synth"],
            load_original_ws=self.opts.w_lambda_psp > 0,
            synth_data_ratio=self.opts.synth_data_ratio,
            use_labels=True,
        )

        test_size = int(len(full_dataset) * 0.05)
        train_size = len(full_dataset) - test_size
        # train_size, test_size = (1.0 - 0.05), 0.05
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def configure_dataloaders(self, batch_size):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(self.opts.workers), drop_last=True
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(self.opts.test_workers),
            drop_last=True,
        )

    def calc_loss_psp(self, x, y_hat, w_predicted, w_original, synth_data, loss_dict):
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda_psp > 0:
            loss_id, sim_improvement, _ = self.id_loss(y_hat, x, x)
            loss_dict["loss_id_psp"] = float(loss_id)
            loss_dict["id_improve_psp"] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda_psp

        if self.opts.l2_lambda_psp > 0:
            loss_l2 = F.mse_loss(y_hat, x)
            loss_dict["loss_l2_psp"] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda_psp

        if self.opts.lpips_lambda_psp > 0:
            loss_lpips = self.lpips_loss(y_hat, x)
            loss_dict["loss_lpips_psp"] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda_psp

        if self.opts.w_lambda_psp > 0 and torch.any(synth_data):
            w_loss = 0
            w_count = 0
            for i in synth_data:
                if not i:
                    continue
                w_count += 1
                loss_w = self.w_loss(w_predicted[i].unsqueeze(0), w_original[i].unsqueeze(0))
                loss_dict["loss_w_psp"] = float(loss_w)
                w_loss += loss_w * self.opts.w_lambda_psp

            loss += w_loss / w_count

        loss_dict["loss_psp"] = float(loss)
        return loss, loss_dict, id_logs

    def calc_mirror_loss_psp(self, x, x_mirror, y_hat, loss_dict):  # removed conf_map
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda_psp_mirror > 0:
            loss_id, sim_improvement, _ = self.id_loss(y_hat, x_mirror, x_mirror)
            loss_dict["loss_id_psp_mirror"] = float(loss_id)
            loss_dict["id_improve_psp_mirror"] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda_psp_mirror

        if self.opts.lpips_lambda_psp_mirror > 0:
            loss_lpips = self.lpips_loss(y_hat, x_mirror)
            loss_dict["loss_lpips_psp_mirror"] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda_psp_mirror

        if self.opts.l2_lambda_psp_mirror > 0:
            # if self.opts.normalize_mirror:
            #     loss_l2 = torch.square(x_mirror - y_hat)
            #     loss_l2 = loss_l2.mean(dim=1)
            #     loss_l2 = loss_l2 / (conf_map + EPS)
            #     loss_l2 = loss_l2.mean(dim=(1, 2))
            #     conf_map_mean = (1 / (conf_map + EPS)).mean(dim=(1, 2))
            #     loss_l2 = (loss_l2 / conf_map_mean).mean()
            # else:

            loss_l2 = torch.square(x_mirror - y_hat)
            loss_l2 = loss_l2.mean(dim=1)

            # if conf_map is not None:
            #     loss_l2 = loss_l2 / (conf_map + EPS)

            loss_l2 = loss_l2.mean()
            loss_dict["loss_l2_psp_mirror"] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda_psp_mirror

        loss_dict["loss_psp_mirror"] = float(loss)
        return loss, loss_dict, id_logs

    def calc_loss_triplane(self, x, y_hat, loss_dict):
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda_triplane > 0:
            loss_id, sim_improvement, _ = self.id_loss(y_hat, x, x)
            loss_dict["loss_id_triplane"] = float(loss_id)
            loss_dict["id_improve_triplane"] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda_triplane

        if self.opts.l1_lambda_triplane > 0:
            loss_l1 = F.smooth_l1_loss(y_hat, x)
            loss_dict["loss_l1_triplane"] = float(loss_l1)
            loss += loss_l1 * self.opts.l1_lambda_triplane
        if self.opts.lpips_lambda_triplane > 0:
            loss_lpips = self.lpips_loss(y_hat, x)
            loss_dict["loss_lpips_triplane"] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda_triplane

        loss_dict["loss_triplane"] = float(loss)
        return loss, loss_dict, id_logs

    def calc_mirror_loss_triplane(self, x, x_mirror, y_hat, loss_dict):
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda_triplane_mirror > 0:
            loss_id, sim_improvement, _ = self.id_loss(y_hat, x_mirror, x_mirror)
            loss_dict["loss_id_triplane_mirror"] = float(loss_id)
            loss_dict["id_improve_triplane_mirror"] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda_triplane_mirror

        if self.opts.lpips_lambda_triplane_mirror > 0:
            loss_lpips = self.lpips_loss(y_hat, x_mirror)
            loss_dict["loss_lpips_triplane_mirror"] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda_triplane_mirror

        if self.opts.l1_lambda_triplane_mirror > 0:
            # if self.opts.normalize_mirror:
            #     loss_l1 = F.smooth_l1_loss(y_hat, x_mirror, reduction="none")
            #     loss_l1 = loss_l1.mean(dim=1)
            #     loss_l1 = loss_l1 / (conf_map + EPS)
            #     loss_l1 = loss_l1.mean(dim=(1, 2))
            #     conf_map_mean = (1 / (conf_map + EPS)).mean(dim=(1, 2))
            #     loss_l1 = (loss_l1 / conf_map_mean).mean()
            # else:

            loss_l1 = F.smooth_l1_loss(y_hat, x_mirror, reduction="none")
            loss_l1 = loss_l1.mean(dim=1)
            # if conf_map is not None:
            #     loss_l1 = loss_l1 / (conf_map + EPS)

            loss_l1 = loss_l1.mean()
            loss_dict["loss_l2_triplane_mirror"] = float(loss_l1)
            loss += loss_l1 * self.opts.l1_lambda_triplane_mirror

        loss_dict["loss_triplane_mirror"] = float(loss)
        return loss, loss_dict, id_logs

    def g_nonsaturating_loss(self, fake_preds):
        loss = F.softplus(-fake_preds).mean()
        return loss

    def calc_discriminator_loss(
        self, loss_dict, generated_triplanes, real_triplanes, generated_occlusion_mask, real_occlusion_mask
    ):
        fake_preds = self.discriminator(generated_triplanes)
        real_preds = self.discriminator(real_triplanes)

        if self.opts.discriminator_name == "TriplaneDiscriminator2DLossMasking":
            assert (
                self.opts.use_occlusion_aware_masks_on_triplanes
            ), "TriplaneDiscriminator2DLossMasking requires occlusion aware masks"

            resized_generated_occlusion_mask = F.interpolate(
                generated_occlusion_mask, size=(32, 32), mode="bilinear", align_corners=False
            )
            resized_real_occlusion_mask = F.interpolate(
                real_occlusion_mask, size=(32, 32), mode="bilinear", align_corners=False
            )
            fake_preds = fake_preds * resized_generated_occlusion_mask
            real_preds = real_preds * resized_real_occlusion_mask

        loss = self.d_logistic_loss(real_preds, fake_preds)
        loss_dict["loss_D"] = float(loss)
        return loss, loss_dict

    def d_logistic_loss(self, real_preds, fake_preds):
        real_loss = F.softplus(-real_preds)
        fake_loss = F.softplus(fake_preds)

        return (real_loss.mean() + fake_loss.mean()) / 2

    def d_r1_loss(self, real_pred, real_triplane):
        (grad_real,) = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_triplane, create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def calc_discriminator_r1_loss(self, loss_dict, real_triplanes):
        real_img_tmp_triplane = real_triplanes.detach().requires_grad_(True)

        real_preds = self.discriminator(real_img_tmp_triplane)
        real_preds = real_preds.view(real_img_tmp_triplane.size(0), -1)
        real_preds = real_preds.mean(dim=1).unsqueeze(1)

        r1_loss = self.d_r1_loss(real_preds, real_img_tmp_triplane)
        loss_D_R1 = self.opts.d_r1_gamma / 2 * r1_loss * self.opts.d_reg_every + 0 * real_preds[0]
        loss_dict["loss_D_r1_reg"] = float(loss_D_R1)
        return loss_D_R1, loss_dict

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f"{prefix}/{key}", value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f"Metrics for {prefix}, step {self.global_step}")
        for key, value in metrics_dict.items():
            print(f"\t{key} = ", value)

    def parse_and_log_images(self, id_logs, x, y_hat, y_hat_psp, title, subscript=None, display_count=2):
        im_data = []
        for i in range(min(self.opts.batch_size, display_count)):
            cur_im_data = {
                "input_face": common.log_input_image(x[i], self.opts),
                "y_hat": common.tensor2im(y_hat[i]),
                "y_hat_psp": common.tensor2im(y_hat_psp[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f"{subscript}_{step:04d}.jpg")
        else:
            path = os.path.join(self.logger.log_dir, name, f"{step:04d}.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            "state_dict": self.net.state_dict(),
            "opts": vars(self.opts),
            "best_val_loss": self.best_val_loss,
            "step": self.global_step,
            "triplane_optimizer": self.optimizer_triplane.state_dict(),
            "psp_optimizer": self.optimizer_psp.state_dict(),
        }

        if self.opts.use_discriminator:
            save_dict["discriminator_state_dict"] = self.discriminator.state_dict()
            save_dict["discriminator_optimizer"] = self.optimizer_discriminator.state_dict()

        return save_dict

    def randomly_sample_camera_poses(self, N):
        sampled_poses = random.sample(self.panohead_cam_list, N)
        sampled_poses = torch.from_numpy(np.array(sampled_poses).reshape(N, -1)).to(self.device).float()
        return sampled_poses

    def get_occluded_triplanes(self, rec_img_dict):
        if not self.opts.use_occlusion_aware_masks_on_triplanes:
            return rec_img_dict["triplanes"], None

        bsz = rec_img_dict["triplanes"].shape[0]

        with torch.no_grad():
            # ray_origins = rec_img_dict["ray_origins"]  # B 64x64  3
            # ray_directions = rec_img_dict["ray_directions"]  # B 64x64  3
            # depth = rec_img_dict["image_depth"].flatten(start_dim=1).reshape(bsz, -1, 1).detach()  # B 64x64  1
            # xyz = ray_origins + depth * ray_directions  # surface xyz       B 64x64  3
            xyz = rec_img_dict["xyz_final"]  # surface xyz B 64 64 3

            x = (xyz[:, :, 0] + 0.5) * 256
            y = (xyz[:, :, 1] + 0.5) * 256
            z = (xyz[:, :, 2] + 0.5) * 256
            x = torch.clamp(x, min=0, max=255)
            y = torch.clamp(y, min=0, max=255)
            z = torch.clamp(z, min=0, max=255)
            x = x.long()
            y = y.long()
            z = z.long()

            # mask = torch.zeros((bsz, 1, 96, 256, 256))
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

                    visible_points_mask[j, k] = self.dilate(visible_points_mask[j, k : k + 1], 5)
                    visible_points_mask[j, k] = self.gauss_kernel_3(visible_points_mask[j, k : k + 1])

        visible_points_mask = visible_points_mask.to(self.device)
        occluded_points_mask = 1 - visible_points_mask

        triplanes = rec_img_dict["triplanes"]
        occluded_triplane = torch.zeros_like(triplanes)

        # B, 96, 256, 256 = B, 1, 256, 256 * B, 96, 256, 256
        occluded_triplane[:, 0, :, :, :] = (occluded_points_mask[:, 0, :, :]).unsqueeze(1) * triplanes[:, 0, :, :, :]
        occluded_triplane[:, 1, :, :, :] = (occluded_points_mask[:, 1, :, :]).unsqueeze(1) * triplanes[:, 1, :, :, :]
        occluded_triplane[:, 2, :, :, :] = (occluded_points_mask[:, 2, :, :]).unsqueeze(1) * triplanes[:, 2, :, :, :]

        return occluded_triplane, visible_points_mask
