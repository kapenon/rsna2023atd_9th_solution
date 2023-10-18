# %%
import os
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path += ["/home/k_takenouchi/workspace/rsna2023_atd/exp"]

data_dir = Path("/home/k_takenouchi/workspace/rsna2023_atd/data/001")
image_dir = Path("/home/k_takenouchi/workspace/rsna2023_atd/data/001/train_images/")

cache_dir = Path("/home/k_takenouchi/workspace/rsna2023_atd/exp/300/302/cache")
cache_dir.mkdir(exist_ok=True, parents=True)
work_dir = Path("./work_dirs")
work_dir.mkdir(exist_ok=True, parents=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# use N * C slices per patient
D = 96
C = 3
image_size = 256
# image_mean = 0.1606
# image_std = 0.2490
n_epochs = 20
p_rand_order = 0.2
lr = 1e-4
eta_min = 1e-5
bs = 4
# accumulate_grad_batches = 3
drop_rate_last = 0
backbone = "seresnext26d_32x4d"

debug = False
use_cache = False

target_name_weight = {
    "bowel_healthy": 1,
    "bowel_injury": 2,
    "extravasation_healthy": 1,
    "extravasation_injury": 6,
    "kidney_healthy": 1,
    "kidney_low": 2,
    "kidney_high": 4,
    "liver_healthy": 1,
    "liver_low": 2,
    "liver_high": 4,
    "spleen_healthy": 1,
    "spleen_low": 2,
    "spleen_high": 4,
    "any_injury": 6,
}
target_names = list(target_name_weight.keys())

target_weights = list(target_name_weight.values())

target_healthy_index = [
    i for i, (name, _) in enumerate(target_name_weight.items()) if "healthy" in name
]

df = pd.read_csv("/home/k_takenouchi/workspace/rsna2023_atd/data/001/split_v2.csv")
df_split = pd.read_csv(
    "/home/k_takenouchi/workspace/rsna2023_atd/data/001/split_v2.csv"
)
df_processed = pd.read_pickle(
    "/home/k_takenouchi/workspace/rsna2023_atd/data/001/processed.pkl"
)
df = df_split.merge(df_processed, how="left", on="series_id", suffixes=["_orig", ""])
df["image_dir"] = str(image_dir) + "/" + df["series_id"].astype(str)
train_meta_df = pd.read_csv(
    "/home/k_takenouchi/workspace/rsna2023_atd/data/000/train_series_meta.csv"
)
train_meta_df = (
    train_meta_df.sort_values(["aortic_hu"])
    .reset_index(drop=True)
    .groupby("patient_id")
    .head(1)
)
use_series_ids = train_meta_df["series_id"].values
df = df[df["series_id"].isin(use_series_ids)].reset_index(drop=True)
# voxel crop用に角座標を追加
df_voxel_crop = pd.read_pickle(
    "/home/k_takenouchi/workspace/rsna2023_atd/exp/300/302/voxel_crop.pkl"
)
df = df.merge(df_voxel_crop, on="series_id")


# segの予測zと、z方向indexとの対応関係の情報取っとくの忘れてた（本来は）
def seg_load_series_images(series_path):
    series_path = Path(series_path)
    t_paths = sorted(series_path.glob("*.png"), key=lambda x: int(x.stem))

    n_scans = len(t_paths)
    indices = (
        np.quantile(list(range(n_scans)), np.linspace(0.0, 1.0, 128))
        .round()
        .astype(int)
    )
    return indices


df["seg_indices"] = df["image_dir"].map(seg_load_series_images)
df["seg_npy_path"] = df["series_id"].map(
    lambda x: f"/home/k_takenouchi/workspace/rsna2023_atd/exp/100/101/predictions/{x}.npy"
)

# ラベルを多クラス用に
binary_targets = ["bowel", "extravasation"]
triple_level_targets = ["liver", "spleen", "kidney"]
targets = binary_targets + triple_level_targets
for t in binary_targets:
    df[t] = df[f"{t}_injury"]
for t in triple_level_targets:
    df[t] = np.argmax(df[[f"{t}_healthy", f"{t}_low", f"{t}_high"]].values, axis=-1)

# liverとspleenとkidneyだけ
df_all = df[df["organ"].isin(triple_level_targets)]

# padding
df_min = df_all.groupby("series_id")[["xmin", "ymin", "zmin"]].agg(min)
df_min[["xmin", "ymin"]] = (df_min[["xmin", "ymin"]] - 0.01).clip(0.0, 1.0)
df_max = df_all.groupby("series_id")[["xmax", "ymax", "zmax"]].agg(max)
df_max[["xmax", "ymax"]] = (df_max[["xmax", "ymax"]] + 0.01).clip(0.0, 1.0)

df_series = df.groupby("series_id").head(1)
df_series = df_series.drop(columns=["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"])
df_series = df_series.merge(df_min, how="left", on="series_id")
df = df_series.merge(df_max, how="left", on="series_id")

# # %% do: label assertion
# df_t = pd.read_csv("/home/k_takenouchi/workspace/rsna2023_atd/data/000/train.csv")
# df_t = df_t.merge(df, how="left", left_on=["patient_id"], right_on=["patient_id_orig"], suffixes=["", "_pp"])
# for row in df_t.itertuples():
#     if row.liver_healthy == 1:
#         assert row.liver == 0
#     if row.liver_low == 1:
#         assert row.liver == 1
#     if row.liver_high == 1:
#         assert row.liver == 2
#     if row.spleen_healthy == 1:
#         assert row.spleen == 0
#     if row.spleen_low == 1:
#         assert row.spleen == 1
#     if row.spleen_high == 1:
#         assert row.spleen == 2
# # %% done: label assertion


# %% dataset
def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_seg_pred(seg_pred_path):
    seg_pred = np.load(seg_pred_path)  # (c, h, w, d)
    tmp = []
    for cid in range(seg_pred.shape[0]):
        tmp += [seg_pred[cid, :, :, :]]
    seg_pred = np.stack(tmp)
    seg_pred = (seg_pred > 0.5).astype(np.uint8)
    for cid in range(seg_pred.shape[0]):
        seg_pred[cid] *= cid + 1
    seg_pred = seg_pred.max(axis=0)

    assert seg_pred.dtype == np.uint8

    return seg_pred  # h, w, d


# # %% do: load_seg_pred()
# seg_pred = load_seg_pred(df["seg_npy_path"].values[0])
# xmin, ymin, xmax, ymax = (df[["xmin", "ymin", "xmax", "ymax"]].values[0] * 128).astype(int)
# zmin, zmax = df[["zmin", "zmax"]].values[0]
# from matplotlib import pyplot as plt

# for i in range(zmin, zmax, 10):
#     img = seg_pred[ymin:ymax, xmin:xmax, i] * 50

#     plt.imshow(img, cmap="gray")
#     plt.show()

# # %% done:

# # %% do: visualize organ length distribution.
# from tqdm import tqdm

# v = []
# for i in tqdm(range(0, 3000, 2)):
#     if np.random.rand() > 0.1:
#         continue
#     seg_pred = load_seg_pred(df["seg_npy_path"].values[i])
#     v += [(((1 <= seg_pred) & (seg_pred <= 3)).sum(axis=0).sum(axis=0) > 10).sum()]

# import seaborn as sns

# sns.displot(v)

# # 0	background
# # 1	liver
# # 2	spleen
# # 3	kidney
# # 4	bowel

# # sns.displot(v[1]) -> 50
# # sns.displot(v[2]) -> 30
# # sns.displot(v[3]) -> 40
# # sns.displot(v[4]) -> 110

# # %% done:


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


# def load_series_images(series_id, n_scans, seg_indices, seg_npy_path, bbox, zmin, zmax):
#     # [zmin, zmax)
#     d = zmax - zmin
#     # use cache
#     cache_file = cache_dir / f"{series_id}.npy"
#     if use_cache and cache_file.exists():
#         return np.load(cache_file)

#     # load segmentation
#     seg_pred = load_seg_pred(seg_npy_path)  # h,w,d
#     seg_pred = ((1 <= seg_pred) & (seg_pred <= 3)).astype(np.uint8)  # to binary
#     xmin, ymin, xmax, ymax = (bbox * seg_pred.shape[0]).astype(np.uint16)
#     # padding
#     seg_pred = seg_pred[ymin:ymax, xmin:xmax]  # crop
#     # seg_pred = cv2.resize(seg_pred, (image_size, image_size), interpolation=cv2.INTER_LINEAR)  # resize

#     images = []
#     c = C // 2
#     for i, seg_index in enumerate(seg_indices):  # iteration回数は変えない
#         # z-crop
#         if i < zmin or zmax <= i:
#             continue
#         tmp = []
#         for di in range(-c, c + 1):
#             j = min(max(seg_index + di, 0), n_scans - 1)
#             img = load_image(image_dir / f"{series_id}" / f"{j}.png")  # load
#             xmin, ymin, xmax, ymax = (bbox * img.shape[0]).astype(np.uint16)
#             img = img[ymin:ymax, xmin:xmax]  # crop
#             img = pad_resize(img, image_size, image_size)
#             # img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)  # resize
#             tmp += [img]
#         # tmp += [seg_pred[:, :, i]]
#         tmp = np.stack(tmp, -1)  # h, w, c
#         images += [tmp]
#     images = np.stack(images)  # d, h, w, c
#     # assert images.shape == (d, image_size, image_size, C + 1)
#     assert images.shape == (d, image_size, image_size, C)

#     use_z = np.quantile(list(range(d)), np.linspace(0, 1, D)).round().astype(int)
#     images = images[use_z]

#     if use_cache:
#         np.save(cache_file, images)

#     # assert images.shape == (D, image_size, image_size, C + 1)
#     assert images.shape == (D, image_size, image_size, C)

#     return images  # (D, h, w, c)


# bbox: 0-1 np.array (xmin, ymin, xmax, ymax)
def load_series_images(series_id, n_scans, seg_indices, seg_npy_path, bbox, zmin, zmax):
    # [zmin, zmax)

    # load segmentation
    seg_pred = load_seg_pred(seg_npy_path)  # h,w,d
    msk = ((1 <= seg_pred) & (seg_pred <= 3)).astype(np.uint8)  # to binary
    seg_pred = seg_pred * msk
    xmin, ymin, xmax, ymax = (bbox * seg_pred.shape[0]).astype(np.uint16)
    # padding
    seg_pred = seg_pred[ymin:ymax, xmin:xmax, zmin:zmax]  # crop
    seg_pred = pad_resize(seg_pred, image_size, image_size)
    # seg_pred = cv2.resize(seg_pred, (image_size, image_size), interpolation=cv2.INTER_LINEAR)  # resize
    seg_pred = seg_pred.transpose(2, 0, 1)  # d, h, w
    # seg_pred = pad_resize(seg_pred, D, image_size)
    seg_pred = cv2.resize(
        seg_pred, (image_size, D), interpolation=cv2.INTER_LINEAR
    )  # resize
    seg_pred = seg_pred.transpose(1, 2, 0)
    assert seg_pred.shape == (image_size, image_size, D)

    use_z = (
        np.quantile(
            list(range(seg_indices[zmin], seg_indices[zmax - 1] + 1)),
            np.linspace(0, 1, D),
        )
        .round()
        .astype(int)
    )

    images = []
    c = C // 2
    for i, zi in enumerate(use_z):  # iteration回数は変えない
        tmp = []
        for di in range(-c, c + 1):
            j = min(max(zi + di, 0), n_scans - 1)
            img = load_image(image_dir / f"{series_id}" / f"{j}.png")  # load
            xmin, ymin, xmax, ymax = (bbox * img.shape[0]).astype(np.uint16)
            img = img[ymin:ymax, xmin:xmax]  # crop
            img = pad_resize(img, image_size, image_size)
            # img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)  # resize
            tmp += [img]
        tmp += [seg_pred[:, :, i]]
        tmp = np.stack(tmp, -1)  # h, w, c
        images += [tmp]
    images = np.stack(images)  # d, h, w, c
    assert images.shape == (D, image_size, image_size, C + 1)
    # assert images.shape == (d, image_size, image_size, C)

    assert images.shape == (D, image_size, image_size, C + 1)
    # assert images.shape == (D, image_size, image_size, C)

    return images  # (D, h, w, c)


# # %% do: load_series_images()
# from matplotlib import pyplot as plt
# from icecream import ic

# series_id = ic(df["series_id"].values[0])
# n = df["n"].values[0]
# seg_indices = df["seg_indices"].values[0]
# seg_npy_path = df["seg_npy_path"].values[0]
# xmin = ic(df["xmin"].values[0])
# ymin = ic(df["ymin"].values[0])
# zmin = ic(df["zmin"].values[0])
# xmax = ic(df["xmax"].values[0])
# ymax = ic(df["ymax"].values[0])
# zmax = ic(df["zmax"].values[0])
# bbox = np.array([xmin, ymin, xmax, ymax])
# images = load_series_images(series_id, n, seg_indices, seg_npy_path, bbox, zmin, zmax)

# for i in range(0, len(images), 50):
#     plt.imshow(images[i, :, :, 0:3])
#     plt.show()
#     plt.imshow(images[i, :, :, -1], cmap="gray")
#     plt.show()

# # %% done:

transforms_train = A.ReplayCompose(
    [
        A.Resize(image_size, image_size),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Transpose(p=0.5),
        # A.RandomBrightness(limit=0.1, p=0.7),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=4, p=0.8
        ),
        # A.OneOf(
        #     [
        #         A.MotionBlur(blur_limit=3),
        #         A.MedianBlur(blur_limit=3),
        #         A.GaussianBlur(blur_limit=3),
        #         # A.GaussNoise(var_limit=(3.0, 9.0)),
        #     ],
        #     p=0.5,
        # ),
        # A.OneOf(
        #     [
        #         A.OpticalDistortion(distort_limit=1.0),
        #         A.GridDistortion(num_steps=5, distort_limit=1.0),
        #     ],
        #     p=0.5,
        # ),
        # A.Cutout(
        #     max_h_size=int(image_size * 0.3),
        #     max_w_size=int(image_size * 0.3),
        #     num_holes=1,
        #     p=0.5,
        # ),
    ]
)

transforms_valid = A.ReplayCompose(
    [
        A.Resize(image_size, image_size),
    ]
)


class CLSDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index()
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        bbox = np.array([row.xmin, row.ymin, row.xmax, row.ymax])
        images = load_series_images(
            row.series_id,
            row.n,
            row.seg_indices,
            row.seg_npy_path,
            bbox,
            row.zmin,
            row.zmax,
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
            # tmp[:C] = (tmp[:C] / 255.0 - image_mean) / image_std  # normalize
            tmp[:C] = tmp[:C] / 255.0
            result += [tmp]

        images = np.stack(result)
        images = torch.tensor(images).float()  # D,c,h,w
        labels = row[triple_level_targets].astype(np.uint8)
        labels = torch.tensor(labels, dtype=torch.long)

        if random.random() < p_rand_order:
            indices = torch.randperm(images.size(0))
            images = images[indices]

        return images, labels


# # %% do: dataset()
# from matplotlib import pyplot as plt

# ds = CLSDataset(df, transforms_train)

# for j, (images, labels) in enumerate(ds):
#     images = images.transpose(1, 3)
#     for i in range(0, len(images), 20):
#         rgb = images[i, :, :, 1:4]
#         plt.imshow((rgb + rgb.min()) / rgb.max())
#         plt.show()
#         # plt.imshow(images[i, :, :, -1])
#         # plt.show()
#     if j == 1:
#         break

# # %% done:


# %%
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # 現在の値（最新）
        self.avg = 0  # 平均値
        self.sum = 0  # 値の合計
        self.count = 0  # 値の数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


target_name_weight = {
    "bowel_healthy": 1,
    "bowel_injury": 2,
    "extravasation_healthy": 1,
    "extravasation_injury": 6,
    "kidney_healthy": 1,
    "kidney_low": 2,
    "kidney_high": 4,
    "liver_healthy": 1,
    "liver_low": 2,
    "liver_high": 4,
    "spleen_healthy": 1,
    "spleen_low": 2,
    "spleen_high": 4,
    "any_injury": 6,
}

# %%
weights = dict(
    bowel=[1, 2],
    extravasation=[1, 6],
    kidney=[1, 2, 4],
    liver=[1, 2, 4],
    spleen=[1, 2, 4],
)

norm = np.linalg.norm(np.stack([np.sum(v) for v in weights.values()]))
loss_weights = {k: np.sum(v) / norm for k, v in weights.items()}


# %%
ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0])).to("cuda")


def criterion(
    logit_liver,
    logit_spleen,
    logit_kidney,
    gt_liver,
    gt_spleen,
    gt_kidney,
    weights=[1, 1, 1],
):
    loss_liver = ce(logit_liver, gt_liver)
    loss_spleen = ce(logit_spleen, gt_spleen)
    loss_kidney = ce(logit_kidney, gt_kidney)
    loss = loss_liver * weights[0] + loss_spleen * weights[1] + loss_kidney * weights[2]
    loss /= sum(weights)
    return loss


# def criterion(logits, targets):
#     """
#     重み付き対数損失を計算する関数

#     Args:
#     - predictions: モデルの予測確率（Tensor）
#     - targets: 真のクラスラベル（Tensor）
#     - class_weights: クラスごとの重み（Tensor）

#     Returns:
#     - weighted_logloss: 重み付き対数損失
#     """
#     predictions = predictions.sigmoid()
#     any_injury, _ = torch.max(1 - predictions[:, target_healthy_index], dim=-1)
#     predictions = torch.concat([predictions, any_injury.view(-1, 1)], dim=-1)

#     # クラスごとの対数損失を計算
#     log_loss = F.binary_cross_entropy(predictions, targets, reduction="none")

#     # クラスごとの対数損失に重みを掛ける
#     target_weights_tensor = torch.tensor(target_weights).to(torch.float32).to("cuda")
#     target_weights_tensor /= torch.norm(target_weights_tensor, p=2)
#     weighted_log_loss = log_loss * target_weights_tensor

#     return weighted_log_loss.mean()


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim
        )

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class CustomModel(pl.LightningModule):
    def __init__(self):
        out_dim = 256
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=C + 1)
        if "resnet" in backbone or "seresnext" in backbone:
            hdim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        if "convnext" in backbone:
            hdim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        print("hdim=", hdim)
        # self.lstm = nn.LSTM(hdim, out_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(
            hdim, out_dim, num_layers=2, bidirectional=True, batch_first=True
        )
        # self.lstm_bn = nn.BatchNorm1d(D)
        self.relu = nn.ReLU()

        self.conv1d_kidney = nn.Conv1d(D, 1, 1)
        self.attn_kidney = Attention(feature_dim=out_dim * 2, step_dim=D)
        self.attn_bn_kidney = nn.BatchNorm1d(out_dim * 2)
        self.head_kidney = nn.Sequential(
            nn.Linear(out_dim * 2 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, 3),
        )
        self.conv1d_liver = nn.Conv1d(D, 1, 1)
        self.attn_liver = Attention(feature_dim=out_dim * 2, step_dim=D)
        self.attn_bn_liver = nn.BatchNorm1d(out_dim * 2)
        self.head_liver = nn.Sequential(
            nn.Linear(out_dim * 2 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, 3),
        )
        self.conv1d_spleen = nn.Conv1d(D, 1, 1)
        self.attn_spleen = Attention(feature_dim=out_dim * 2, step_dim=D)
        self.attn_bn_spleen = nn.BatchNorm1d(out_dim * 2)
        self.head_spleen = nn.Sequential(
            nn.Linear(out_dim * 2 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, 3),
        )

    def forward(self, x):
        b, d, c, h, w = x.shape
        x = x.contiguous().view(b * d, c, h, w)
        x = self.backbone(x)
        x = x.view(b, d, -1)  # b,d,c
        # lstm
        x, _ = self.lstm(x)
        # x = self.lstm_bn(x)
        # x = self.relu(x)

        # 1.liver
        x_conv_liver = self.conv1d_liver(x)[:, 0]
        x_attn_liver = self.attn_liver(x)
        x_attn_liver = self.attn_bn_liver(x_attn_liver)
        x_attn_liver = self.relu(x_attn_liver)
        x_liver = torch.cat([x_conv_liver, x_attn_liver], dim=-1)
        logit_liver = self.head_liver(x_liver)

        # 2.spleen
        x_conv_spleen = self.conv1d_spleen(x)[:, 0]
        x_attn_spleen = self.attn_spleen(x)
        x_attn_spleen = self.attn_bn_spleen(x_attn_spleen)
        x_attn_spleen = self.relu(x_attn_spleen)
        x_spleen = torch.cat([x_conv_spleen, x_attn_spleen], dim=-1)
        logit_spleen = self.head_spleen(x_spleen)

        # 3.kidney
        x_conv_kidney = self.conv1d_kidney(x)[:, 0]
        x_attn_kidney = self.attn_kidney(x)
        x_attn_kidney = self.attn_bn_kidney(x_attn_kidney)
        x_attn_kidney = self.relu(x_attn_kidney)
        x_kidney = torch.cat([x_conv_kidney, x_attn_kidney], dim=-1)
        logit_kidney = self.head_kidney(x_kidney)
        return logit_liver, logit_spleen, logit_kidney  # b,n_organ,n_target

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=eta_min
        )  # T_max is the number of epochs
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        self.meter_train_loss = AverageMeter()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        gt_liver = labels[:, 0]
        gt_spleen = labels[:, 1]
        gt_kidney = labels[:, 2]
        logit_liver, logit_spleen, logit_kidney = self.forward(inputs)
        loss = criterion(
            logit_liver=logit_liver,
            logit_spleen=logit_spleen,
            logit_kidney=logit_kidney,
            gt_liver=gt_liver,
            gt_spleen=gt_spleen,
            gt_kidney=gt_kidney,
        )
        self.meter_train_loss.update(loss.item())
        self.log("train_loss", self.meter_train_loss.avg)
        return loss

    def on_validation_epoch_start(self):
        self.meter_val_loss = AverageMeter()
        self.labels = []
        self.logits_liver = []
        self.logits_spleen = []
        self.logits_kidney = []

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = batch
        gt_liver = labels[:, 0]
        gt_spleen = labels[:, 1]
        gt_kidney = labels[:, 2]
        logit_liver, logit_spleen, logit_kidney = self.forward(inputs)
        loss = criterion(
            logit_liver=logit_liver,
            logit_spleen=logit_spleen,
            logit_kidney=logit_kidney,
            gt_liver=gt_liver,
            gt_spleen=gt_spleen,
            gt_kidney=gt_kidney,
        )
        self.meter_val_loss.update(loss.item())

        self.labels += [labels.cpu()]
        self.logits_liver += [logit_liver.cpu()]
        self.logits_spleen += [logit_spleen.cpu()]
        self.logits_kidney += [logit_kidney.cpu()]

    def on_validation_epoch_end(self):
        logits_liver = torch.cat(self.logits_liver, axis=0)  # n, 3
        preds_liver = torch.softmax(logits_liver.float(), dim=-1).numpy()
        logits_spleen = torch.cat(self.logits_spleen, axis=0)  # n, 3
        preds_spleen = torch.softmax(logits_spleen.float(), dim=-1).numpy()
        logits_kidney = torch.cat(self.logits_kidney, axis=0)  # n, 3
        preds_kidney = torch.softmax(logits_kidney.float(), dim=-1).numpy()
        labels = torch.cat(self.labels, axis=0).numpy()  # n, 3

        np.save(
            work_dir / f"logits_liver_epoch{self.current_epoch}.npy",
            logits_liver.numpy(),
        )
        np.save(
            work_dir / f"logits_spleen_epoch{self.current_epoch}.npy",
            logits_spleen.numpy(),
        )
        np.save(
            work_dir / f"logits_kidney_epoch{self.current_epoch}.npy",
            logits_kidney.numpy(),
        )
        np.save(work_dir / "labels.npy", labels)

        scores = 0
        for i, tmp_preds in enumerate((preds_liver, preds_spleen, preds_kidney)):
            tmp_labels = labels[:, i]
            weight_matrix = np.array([[1, 2, 4]] * len(labels))
            sample_weight = weight_matrix[np.arange(weight_matrix.shape[0]), tmp_labels]

            labels_onehot = F.one_hot(torch.tensor(tmp_labels), num_classes=3)
            tmp_score = sklearn.metrics.log_loss(
                y_true=labels_onehot, y_pred=tmp_preds, sample_weight=sample_weight
            )
            self.log(f"score_{triple_level_targets[i]}", tmp_score)
            scores += tmp_score
        self.log("val_loss", self.meter_val_loss.avg)
        self.log("val_score", scores / 3)


# %%
df_train = df[df["fold"] != 0].reset_index(drop=True)
df_valid = df[df["fold"] == 0].reset_index(drop=True)

if debug:
    df_train = df_train.head(bs * 3 - 1)
    df_valid = df_valid.head(bs * 3 - 1)

ds_train = CLSDataset(df_train, transforms_train)
ds_valid = CLSDataset(df_valid, transforms_valid)

dl_train = DataLoader(
    ds_train, batch_size=bs, shuffle=True, drop_last=True, num_workers=4
)
dl_valid = DataLoader(ds_valid, batch_size=bs, num_workers=4)

# Lightningモデルとトレーナーの設定
model = CustomModel()

# # %% do:
# images, labels = iter(dl_train).__next__()
# out = model(images)

# criterion = criterion.cpu()
# criterion(out.cpu(), labels.cpu())


# # %% done:
# %%
callbacks = []
# early_stop_callback = EarlyStopping(
#     monitor="val_loss",
#     patience=3,
#     verbose=True,
#     mode="min",
# )
callbacks += [
    ModelCheckpoint(
        monitor="val_score",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )
]
callbacks += [LearningRateMonitor(logging_interval="step")]

wandb_logger = WandbLogger(
    project="rsna2023_atd",
    name=Path(__file__).resolve().parent.name,
    config=dict(
        D=D,
        C=C,
        image_size=image_size,
        n_epochs=n_epochs,
        p_rand_order=p_rand_order,
        lr=lr,
        bs=bs,
        # accumulate_grad_batches=accumulate_grad_batches,
        eta_min=eta_min,
        drop_rate_last=drop_rate_last,
        backbone=backbone,
    ),
)

trainer = pl.Trainer(
    max_epochs=2 if debug else n_epochs,
    gpus=1,
    devices=[1],
    callbacks=None if debug else callbacks,
    logger=None if debug else wandb_logger,
    log_every_n_steps=1 if debug else 50,
    # accumulate_grad_batches=accumulate_grad_batches,
    # precision=16,
)

# トレーニングの実行
# trainer.fit(model, dl_train, dl_valid)
