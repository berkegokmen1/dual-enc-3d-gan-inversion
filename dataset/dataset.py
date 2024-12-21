# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch

# import dnnlib
import time
from PIL import Image, ImageOps
import torchvision.transforms.functional as F

try:
    import pyspng
except ImportError:
    pyspng = None


# ----------------------------------------------------------------------------


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,  # Name of the dataset.
        raw_shape,  # Shape of the raw image data (NCHW).
        max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
        xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed=0,  # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._raw_labels_synth = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels()  # if self._use_labels else None

            self._raw_labels_labels = [
                self._raw_labels.get(
                    fname.replace("\\", "/").replace(".jpg", ".png"), fname.replace("\\", "/").replace(".png", ".jpg")
                )
                for fname in self._image_fnames
            ]
            self._raw_labels_labels = np.array(self._raw_labels_labels)
            self._raw_labels_labels = self._raw_labels_labels.astype(
                {1: np.int64, 2: np.float32}[self._raw_labels_labels.ndim]
            )

            # if self._raw_labels is None:
            #     self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)

            # assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels_labels.dtype in [np.float32, np.int64]

            if self._raw_labels_labels.dtype == np.int64:
                assert self._raw_labels_labels.ndim == 1
                assert np.all(self._raw_labels_labels >= 0)

            self._raw_labels_std = self._raw_labels_labels.std(0)

        return self._raw_labels

    def _get_raw_labels_for_synth_data(self):
        if self._raw_labels_synth is None:
            self._raw_labels_synth = self._load_raw_labels_for_synth_data()  # if self._use_labels else None

        return self._raw_labels_synth

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_image_for_synth_data(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels_for_synth_data(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image, fname, w_original, synth_data = self._load_raw_image(idx)

        mirror_image = np.array(ImageOps.mirror(Image.fromarray(image.transpose(1, 2, 0)))).transpose(2, 0, 1)

        label = self.get_label(fname, synth_data)
        label = torch.from_numpy(label).float()
        w_original = torch.from_numpy(w_original).float()  # this may have remained as np.zeros((14, 512))

        pose, intrinsics = np.array(label[:16]).reshape(4, 4), np.array(label[16:]).reshape(3, 3)
        flipped_pose = flip_yaw(pose)
        mirror_label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)])
        mirror_label = torch.from_numpy(mirror_label).float()

        assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape  # TODO close due to 563x563 images...
        assert image.dtype == np.uint8

        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]

        image = torch.from_numpy(image).float() / 255
        mirror_image = torch.from_numpy(mirror_image).float() / 255

        image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        mirror_image = F.normalize(mirror_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        image = F.resize(image, self.image_shape[1:])
        image_resized = F.resize(image, [256, 256])

        mirror_image = F.resize(mirror_image, self.image_shape[1:])
        mirror_image_resized = F.resize(mirror_image, [256, 256])

        return (
            image_resized,
            image,
            label,
            mirror_image_resized,
            mirror_image,
            mirror_label,
            w_original,
            fname,
            synth_data,
        )

    def get_label(self, fname, synth_data):
        label = (
            self._get_raw_labels().get(fname.replace(".jpg", ".png"), fname.replace(".png", ".jpg"))
            if not synth_data
            else self._get_raw_labels_for_synth_data().get(
                fname.replace(".jpg", ".png"), self._get_raw_labels_for_synth_data().get(fname.replace(".png", ".jpg"))
            )
        )
        label = np.array(label)

        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot

        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()

        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = int(self._xflip[idx]) != 0
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()

        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]

        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        synth_path=None,  # Path to directory or zip with synthetic data.
        resolution=None,  # Ensure specific resolution, None = highest available.
        load_original_ws=False,  # Load original W for synthetic data.
        synth_data_ratio=0.1,  # Ratio of using synthetic data.
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._synth_path = synth_path
        self._zipfile = None
        self._load_original_ws = load_original_ws
        self._synth_data_ratio = synth_data_ratio

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }

        elif self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        PIL.Image.init()

        ## self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        json_path = (
            "/FFHQ_LPFF_rebalanced.json" if os.path.exists(path + "/FFHQ_LPFF_rebalanced.json") else "/dataset.json"
        )

        with open(path + json_path, "r") as f:
            self._image_fnames = json.load(f)

            if json_path == "/dataset.json":
                self._image_fnames = [x[0] for x in self._image_fnames["labels"]]

            self._image_fnames = [fname.replace(".png", ".jpg") for fname in self._image_fnames]
            self._image_list = set(os.listdir(path))
            self._image_fnames = [fname for fname in self._image_fnames if fname in self._image_list]
            self._image_fnames = sorted(self._image_fnames)  # NOTE batu ekledi

        if synth_path is not None:
            with open(synth_path + "/dataset.json", "r") as f:
                self._synth_image_fnames = json.load(f)
                self._synth_image_fnames = map(
                    lambda x: x.replace("png", "jpg"),
                    list(
                        dict(self._synth_image_fnames["labels"]).keys()
                    ),  # TODO map yeni ekledim, mead dosyaları bulamıyordu - berke
                    # NOTE cok iyi oldu, self._synth_data_ratio'u 0 iken real data kısmında png/jpeg sorunu da gidiyor, bir alternatif - batu
                )
                self._synth_image_fnames = [
                    fname for fname in self._synth_image_fnames if fname.endswith(".png") or fname.endswith(".jpg")
                ]
                self._synth_image_list = set(os.listdir(synth_path))
                self._synth_image_fnames = [
                    fname for fname in self._synth_image_fnames if fname in self._synth_image_list
                ]
                self._synth_image_fnames = sorted(self._synth_image_fnames)  # NOTE batu ekledi

        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        if len(self._synth_image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]

        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape)

        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        np.random.seed((int(time.time()) % 1000000) + raw_idx)  # set a different seed
        choice = np.random.rand()
        np.random.seed(2107)  # reset seed

        try:
            if choice < self._synth_data_ratio and self._synth_path is not None:
                return self._load_raw_image_for_synth_data(raw_idx)
            else:
                return self._load_raw_image_for_real_data(raw_idx)

        except Exception as e:
            print(f"Error loading image, loading another -real- image {(raw_idx + 1)}: ", e)
            return (
                self._load_raw_image_for_synth_data(raw_idx + 1)
                if self._synth_data_ratio == 1
                else self._load_raw_image_for_real_data(raw_idx + 1)
            )

    def _load_raw_image_for_real_data(self, raw_idx):
        fname = self._image_fnames[raw_idx % len(self._image_fnames)]

        image = np.array(PIL.Image.open(os.path.join(self._path, fname.replace(".png", ".jpg"))))
        # image = np.array(PIL.Image.open(os.path.join(self._path, fname)))

        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC

        if image.shape[2] == 4:
            image = image[:, :, :3]

        image = image.transpose(2, 0, 1)  # HWC => CHW

        # last-1 argument is None (original W)
        # last argument is False if real data
        return image, fname, np.zeros((14, 512)), False

    def _load_raw_image_for_synth_data(self, raw_idx):
        fname = self._synth_image_fnames[raw_idx % len(self._synth_image_fnames)]

        image = np.array(PIL.Image.open(os.path.join(self._synth_path, fname)))

        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC

        image = image.transpose(2, 0, 1)  # HWC => CHW

        w_original = self._load_original_w_for_synth_data(fname) if self._load_original_ws else np.zeros((14, 512))

        # last-1 argument is original W
        # last argument is True if synthetic data
        return image, fname, w_original, True

    def _load_raw_labels(self):
        fname = "dataset.json"

        if fname not in self._all_fnames:
            return None

        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]

        if labels is None:
            return None

        labels = dict(labels)
        # labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        # labels = np.array(labels)
        # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        return labels

    def _load_raw_labels_for_synth_data(self):
        fname = "dataset.json"

        with open(os.path.join(self._synth_path, fname)) as f:
            labels = json.load(f)["labels"]

        if labels is None:
            return None

        labels = dict(labels)

        return labels

    def _load_original_w_for_synth_data(self, fname):
        fname = fname.replace(".png", ".npy")
        fname = fname.replace(".jpg", ".npy")
        fname = os.path.join(self._synth_path, fname)

        w_original = np.load(fname)

        return w_original

    def __len__(self):
        return len(self._image_fnames)


# ----------------------------------------------------------------------------
