import os
import random
from collections import defaultdict
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
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# dirs
root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
image_dir = data_dir / "png_images"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

D = 96
C = 5
image_size = 512
n_epochs = 15
p_rand_order = 0.2
lr = 8e-4
eta_min = 1e-5
bs = 32
drop_rate_last = 0
backbone = "seresnext26d_32x4d"

df_split = pd.read_csv(data_dir / "split.csv")
df_slice = pd.read_pickle(data_dir / "slice.pkl")
df = df_slice.merge(df_split[["patient_id", "fold"]], how="left", on="patient_id")
df["image_dir"] = str(image_dir) + "/" + df["series_id"].astype(str)
df = df.sort_values(["aortic_hu"]).reset_index(drop=True).groupby("patient_id").tail(1)
df_voxel_crop = pd.read_pickle(data_dir / "organ_voxel.pkl")
df = df.merge(df_voxel_crop, on="series_id")


def seg_load_series_images(series_path):
    series_path = Path(series_path)
    t_paths = sorted(series_path.glob("*.png"), key=lambda x: int(x.stem))

    n_scans = len(t_paths)
    indices = (
        np.quantile(list(range(n_scans)), np.linspace(0.0, 1.0, 128))
        .round()
        .astype(int)
    )
    return indices, n_scans


indices_list = []
n_list = []
for spath in df["image_dir"].values:
    indices, n = seg_load_series_images(spath)
    indices_list += [indices]
    n_list += [n]

df["seg_indices"] = indices_list
df["n"] = n_list
df["seg_npy_path"] = df["series_id"].map(lambda x: data_dir / "seg_pred" / f"{x}.npy")

binary_targets = ["bowel", "extravasation"]
triple_level_targets = ["liver", "spleen", "kidney"]
targets = binary_targets + triple_level_targets
for t in binary_targets:
    df[t] = df[f"{t}_injury"]
for t in triple_level_targets:
    df[t] = np.argmax(df[[f"{t}_healthy", f"{t}_low", f"{t}_high"]].values, axis=-1)

df = df[df["organ"].isin(["bowel"])]
pad = 0.01
df_min = df.groupby("series_id")[["xmin", "ymin", "zmin"]].agg(min)
df_min[["xmin", "ymin"]] = (df_min[["xmin", "ymin"]] / 128.0 - pad).clip(0.0, 1.0)
df_max = df.groupby("series_id")[["xmax", "ymax", "zmax"]].agg(max)
df_max[["xmax", "ymax"]] = (df_max[["xmax", "ymax"]] / 128.0 + pad).clip(0.0, 1.0)

df_series = df.groupby("series_id").head(1)
df_series = df_series.drop(columns=["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"])
df_series = df_series.merge(df_min, how="left", on="series_id")
df = df_series.merge(df_max, how="left", on="series_id")


df_slice = defaultdict(list)
for row in tqdm(df.itertuples()):
    min_slice_id = row.seg_indices[row.zmin]
    max_slice_id = row.seg_indices[row.zmax - 1]

    slice_ids = (
        np.quantile(list(range(row.n)), np.linspace(0.0, 1.0, D)).round().astype(int)
    )

    positive_slice_ids_extravasation = set(row.positive_slice_id_extravasation)
    positive_slice_ids_bowel = set(row.positive_slice_id_bowel)
    label_images = []
    # id of slices
    c = C // 2
    for center_slice_id in slice_ids:
        slice_ids_in_img = [
            min(max(center_slice_id + delta, 0), row.n - 1)
            for delta in range(-c, c + 1)
        ]

        intersect_extravasation = set(slice_ids_in_img).intersection(
            positive_slice_ids_extravasation
        )
        label_img_extravasation = int(len(intersect_extravasation) > 0)

        intersect_bowel = set(slice_ids_in_img).intersection(positive_slice_ids_bowel)
        label_img_bowel = int(len(intersect_bowel) > 0)

        df_slice["slice_ids_in_img"] += [slice_ids_in_img]
        df_slice["series_id"] += [row.series_id]
        df_slice["label_img_extravasation"] += [label_img_extravasation]
        df_slice["label_img_bowel"] += [label_img_bowel]
        df_slice["fold"] += [row.fold]
        df_slice["label_patient_extravasation"] += [row.extravasation]
        df_slice["label_patient_bowel"] += [row.bowel]
        df_slice["aortic_hu"] += [row.aortic_hu]


df_slice = pd.DataFrame(df_slice)
df_patient_pos = df_slice[df_slice["label_patient_extravasation"] == 1]
df_patient_neg = df_slice[df_slice["label_patient_extravasation"] == 0]
df_use = df_patient_pos[df_patient_pos["label_img_extravasation"] > 0]
n_pos = len(df_use)

df_add = df_patient_pos[df_patient_pos["label_img_extravasation"] == 0].sample(
    len(df_use) * 3
)
df_use = pd.concat([df_use, df_add], axis=0).reset_index(drop=True)

n_neg = len(df_use) - n_pos
n_add = max(n_pos * 10 - n_neg, 0)
print(n_pos, n_neg, n_add)
df_add = df_patient_neg[
    (df_patient_neg["label_img_extravasation"] == 0)
    & (df_patient_neg["aortic_hu"] > 200)
].sample(n_add)
df_slice = pd.concat([df_use, df_add], axis=0).reset_index(drop=True)
df_slice["label_img"] = df_slice["label_img_extravasation"]


def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return image


def load_input(series_id, slice_id_list):
    input = [
        load_image(image_dir / f"{series_id}" / f"{slice_id}.png")
        for slice_id in slice_id_list
    ]
    input = np.stack(input, -1)
    assert input.shape == (image_size, image_size, C)
    return input


transforms_train = A.ReplayCompose(
    [
        A.Resize(image_size, image_size),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=4, p=0.8
        ),
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

        inputs = load_input(row.series_id, row.slice_ids_in_img)
        inputs = (
            self.transform(image=inputs)["image"].transpose(2, 0, 1).astype(np.float64)
        )
        inputs /= 255.0
        labels = row.label_img

        inputs = torch.tensor(inputs).float()
        labels = torch.tensor(labels).float()

        return inputs, labels


bce = nn.BCEWithLogitsLoss(reduction="none")


def criterion(logits, targets):
    w = 10.0
    losses = bce(logits.view(-1), targets.view(-1))
    losses[targets.view(-1) > 0] *= w
    norm = torch.ones(logits.view(-1).shape[0]).to("cuda")
    norm[targets.view(-1) > 0] *= w
    return losses.sum() / norm.sum()


class CustomModel(pl.LightningModule):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=C)
        if "resnet" in backbone or "resnext" in backbone:
            hdim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        if "convnext" in backbone:
            hdim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(hdim, hdim // 2),
            nn.BatchNorm1d(hdim // 2),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(hdim // 2, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        logits = self.head(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=eta_min
        )  # T_max is the number of epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = criterion(logits, labels)

        self.log("train_loss", loss.item())
        return loss

    def on_validation_epoch_start(self):
        self.meter_val_loss = AverageMeter()
        self.logits = []
        self.labels = []

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = criterion(logits, labels)
        self.meter_val_loss.update(loss.item())

        self.logits += [logits.cpu()]
        self.labels += [labels.cpu()]

    def on_validation_epoch_end(self):
        logits = torch.cat(self.logits, axis=0)
        preds = torch.sigmoid(logits.float()).numpy().flatten().astype(float)
        labels = torch.cat(self.labels, axis=0).numpy().flatten().astype(int)  # n

        try:
            auc = sklearn.metrics.roc_auc_score(labels, preds)
        except:
            auc = 0
        self.log("val_auc", auc)


def run(fold):
    df_train = df_slice[df_slice["fold"] != fold].reset_index(drop=True)
    df_valid = df_slice[df_slice["fold"] == fold].reset_index(drop=True)

    ds_train = CLSDataset(df_train, transforms_train)
    ds_valid = CLSDataset(df_valid, transforms_valid)

    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=4)
    dl_valid = DataLoader(ds_valid, batch_size=bs, num_workers=4)

    model = CustomModel()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        filename="epoch-{epoch:02d}-{val_auc:.3f}",
        save_top_k=3,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        gpus=1,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50,
    )

    trainer.fit(model, dl_train, dl_valid)


def save_backbone(ckpt, out):
    trained = CustomModel.load_from_checkpoint(ckpt)
    torch.save(
        trained.backbone.state_dict(),
        out,
    )


if __name__ == "__main__":
    for fold in range(4):
        run(fold)

    # Provide the path to the best checkpoint once the training is completed.
    # Save the backbone weight for the second stage.
    checkpoint = ""
    out = "backbone.pth"
    save_backbone(checkpoint, out)
