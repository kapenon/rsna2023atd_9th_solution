import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

triple_level_targets = ["liver", "spleen", "kidney"]


def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def pad_resize(image, h, w):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > w / h:
        new_width = w
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = h
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    top_pad = (h - new_height) // 2
    bottom_pad = h - new_height - top_pad
    left_pad = (w - new_width) // 2
    right_pad = w - new_width - left_pad

    padded_image = cv2.copyMakeBorder(
        resized_image,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    return padded_image


def load_seg_pred(seg_pred_path):
    seg_pred = np.load(seg_pred_path)  # c, h, w, d
    tmp = []
    for cid in range(seg_pred.shape[0]):
        tmp += [seg_pred[cid, :, :, :]]
    seg_pred = np.stack(tmp)
    seg_pred = (seg_pred > 0.5).astype(np.uint8)
    for cid in range(seg_pred.shape[0]):
        seg_pred[cid] *= cid + 1
    seg_pred = seg_pred.max(axis=0)

    return seg_pred  # h, w, d


def load_series_images(
    series_images_dir,
    n_slices,
    seg_slice_ids,
    seg_npy_path,
    bbox,
    zmin,
    zmax,
    image_sizes,
    n_channels,
):
    # load segmentation
    seg_pred = load_seg_pred(seg_npy_path)  # h,w,d
    msk = ((1 <= seg_pred) & (seg_pred <= 3)).astype(np.uint8)  # to binary
    seg_pred = seg_pred * msk
    xmin, ymin, xmax, ymax = (bbox * seg_pred.shape[0]).astype(np.uint16)
    seg_pred = seg_pred[ymin:ymax, xmin:xmax, zmin:zmax]  # crop
    seg_pred = pad_resize(seg_pred, image_sizes[1], image_sizes[1])
    seg_pred = seg_pred.transpose(2, 0, 1)  # d, h, w
    seg_pred = cv2.resize(
        seg_pred, (image_sizes[1], image_sizes[0]), interpolation=cv2.INTER_LINEAR
    )  # resize
    seg_pred = seg_pred.transpose(1, 2, 0)
    assert seg_pred.shape == (image_sizes[1], image_sizes[1], image_sizes[0])

    use_z = (
        np.quantile(
            list(range(seg_slice_ids[zmin], seg_slice_ids[zmax - 1] + 1)),
            np.linspace(0, 1, image_sizes[0]),
        )
        .round()
        .astype(int)
    )

    images = []
    c = n_channels // 2
    for i, zi in enumerate(use_z):
        tmp = []
        for di in range(-c, c + 1):
            j = min(max(zi + di, 0), n_slices - 1)
            img = load_image(Path(series_images_dir) / f"{j}.png")  # load
            xmin, ymin, xmax, ymax = (bbox * img.shape[0]).astype(np.uint16)
            img = img[ymin:ymax, xmin:xmax]  # crop
            img = pad_resize(img, image_sizes[1], image_sizes[2])
            tmp += [img]
        tmp += [seg_pred[:, :, i]]
        tmp = np.stack(tmp, -1)  # h, w, c
        images += [tmp]
    images = np.stack(images)  # d, h, w, c

    return images  # D, h, w, c


class CLSDataset(Dataset):
    def __init__(self, df, transform, p_rand_order, n_channels, image_sizes):
        self.df = df.reset_index()
        self.transform = transform
        self.p_rand_order = p_rand_order
        self.n_channels = n_channels
        self.image_sizes = image_sizes

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        bbox = np.array([row.xmin, row.ymin, row.xmax, row.ymax])
        images = load_series_images(
            row.image_dir,
            row.n_slices,
            row.seg_slice_ids,
            row.seg_npy_path,
            bbox,
            row.zmin,
            row.zmax,
            self.image_sizes,
            self.n_channels,
        )  # d,h,w,c
        result = []
        data = None
        for image in images:
            if data is None:
                data = self.transform(image=image)
                tmp = data["image"].transpose(2, 0, 1).astype(np.float64)  # c,h,w
            else:
                tmp = (
                    self.transform.replay(data["replay"], image=image)["image"]
                    .transpose(2, 0, 1)
                    .astype(np.float64)
                )  # c,h,w
            tmp[: self.n_channels] = tmp[: self.n_channels] / 255.0
            result += [tmp]

        images = np.stack(result)
        images = torch.tensor(images).float()  # D,c,h,w
        labels = row[triple_level_targets].astype(np.uint8)
        labels = torch.tensor(labels, dtype=torch.long)

        if random.random() < self.p_rand_order:
            indices = torch.randperm(images.size(0))
            images = images[indices]

        return images, labels
