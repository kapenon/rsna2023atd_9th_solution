import os
import random
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
C = 4
image_sizes = (384, 384)
n_epochs = 12
p_rand_order = 0
lr = 2e-4
eta_min = 1e-5
bs = 8
drop_rate_last = 0
backbone = "seresnext26d_32x4d"


# dirs
root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
image_dir = data_dir / "png_images"

df_split = pd.read_csv(data_dir / "split.csv")
df = pd.read_pickle(data_dir / "slice.pkl")
df = df.merge(df_split[["patient_id", "fold"]], how="left", on="patient_id")
df = df.sort_values("aortic_hu").groupby("patient_id").head(1).reset_index(drop=True)
df_il = pd.read_csv(data_dir / "bowel_image_level.csv").drop(columns=["fold"])
df = df.merge(df_il, how="left", left_on="patient_id", right_on="pid")
df = df.groupby("sid")[["img_path", "plabel", "fold", "bowel_injury"]].agg(list)
df["plabel"] = df["plabel"].map(lambda x: x[0])
df["fold"] = df["fold"].map(lambda x: x[0])
df["bowel_injury"] = df["bowel_injury"].map(lambda x: x[0])


def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    return image


transforms_train = A.ReplayCompose(
    [
        A.ShiftScaleRotate(
            shift_limit=0, scale_limit=0.1, rotate_limit=10, border_mode=4, p=0.5
        ),
    ]
)

transforms_valid = A.ReplayCompose([])


def resize_d(images):  # d, c, h, w
    d, c, h, w = images.shape
    # d, h, w
    res = [cv2.resize(images[:, ci, :, :], (h, D)) for ci in range(c)]
    return np.stack(res, axis=1)


class CLSDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index()
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = np.stack([load_image(img_path) for img_path in row.img_path])

        # (d, h, w, c)
        result = []
        data = None
        for d in range(inputs.shape[0]):
            if data is None:
                data = self.transform(image=inputs[d])
                tmp = data["image"].transpose(2, 0, 1).astype(np.float64)  # c,h,w
            else:
                tmp = (
                    self.transform.replay(data["replay"], image=inputs[d])["image"]
                    .transpose(2, 0, 1)
                    .astype(np.float64)
                )  # c,h,w
            tmp = tmp / 255.0
            result += [tmp]

        inputs = np.stack(result)
        inputs = resize_d(inputs)
        inputs = torch.tensor(inputs).float()  # D,c,h,w
        labels = row["bowel_injury"]
        labels = torch.tensor(labels).long()

        if random.random() < p_rand_order:
            indices = torch.randperm(inputs.size(0))
            inputs = inputs[indices]

        return inputs, labels


bce = nn.BCEWithLogitsLoss(reduction="none").to("cuda")


def criterion(logits, targets):
    w = 3.0
    losses = bce(logits.view(-1), targets.view(-1))
    losses[targets.view(-1) > 0] *= w
    norm = torch.ones(logits.view(-1).shape[0]).to("cuda")
    norm[targets.view(-1) > 0] *= w
    return losses.sum() / norm.sum()


class CustomModel(pl.LightningModule):
    def __init__(self):
        out_dim = 256
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=C)
        if "resnet" in backbone or "seresnext" in backbone:
            hdim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        if "convnext" in backbone:
            hdim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        print("hdim=", hdim)
        self.lstm = nn.LSTM(
            hdim, out_dim, num_layers=2, bidirectional=True, batch_first=True
        )
        self.relu = nn.ReLU()

        self.conv1d = nn.Conv1d(D, 1, 1)
        self.attn = Attention(feature_dim=out_dim * 2, step_dim=D)
        self.attn_bn = nn.BatchNorm1d(out_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 2 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x):
        b, d, c, h, w = x.shape
        x = x.contiguous().view(b * d, c, h, w)
        x = self.backbone(x)
        x = x.view(b, d, -1)  # b,d,c

        # lstm
        x, _ = self.lstm(x)

        x_conv = self.conv1d(x)[:, 0]
        x_attn = self.attn(x)
        x_attn = self.attn_bn(x_attn)
        x_attn = self.relu(x_attn)
        x = torch.cat([x_conv, x_attn], dim=-1)
        logit = self.head(x)

        return logit  # b,n_organ,n_target

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
        logit = self.forward(inputs).flatten()
        loss = criterion(logit, labels.float())
        self.meter_train_loss.update(loss.item())
        self.log("train_loss", self.meter_train_loss.avg)
        return loss

    def on_validation_epoch_start(self):
        self.meter_val_loss = AverageMeter()
        self.labels = []
        self.logits = []

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logit = self.forward(inputs).flatten()
        loss = criterion(logit, labels.float())
        self.meter_val_loss.update(loss.item())
        self.log("train_loss", self.meter_val_loss.avg)
        self.labels += [labels.cpu()]
        self.logits += [logit.cpu()]

    def on_validation_epoch_end(self):
        logits = torch.cat(self.logits, axis=0)  # n, 3
        preds = torch.sigmoid(logits.float()).numpy()
        labels = torch.cat(self.labels, axis=0).numpy()  # n, 3

        weight_matrix = np.array([[1, 2]] * len(labels))
        sample_weight = weight_matrix[np.arange(weight_matrix.shape[0]), labels]

        if labels.sum() == 0:
            auc = 0
            score = 0
        else:
            score = sklearn.metrics.log_loss(
                y_true=labels, y_pred=preds, sample_weight=sample_weight
            )
            auc = sklearn.metrics.roc_auc_score(
                y_true=labels.flatten(), y_score=preds.flatten()
            )

        self.log("val_loss", self.meter_val_loss.avg)
        self.log("val_score", score)
        self.log("val_auc", auc)


def run(fold, backbone_pth):
    df_train = df[df["fold"] != fold].reset_index(drop=True)
    print(f"== df_train fold{fold} ==")
    print(df_train.groupby("bowel_injury").size())
    df_valid = df[df["fold"] == fold].reset_index(drop=True)
    print(f"== df_valid fold{fold} ==")
    print(df_valid.groupby("bowel_injury").size())

    ds_train = CLSDataset(df_train, transforms_train)
    ds_valid = CLSDataset(df_valid, transforms_valid)

    dl_train = DataLoader(
        ds_train, batch_size=bs, shuffle=True, drop_last=True, num_workers=8
    )
    dl_valid = DataLoader(ds_valid, batch_size=bs, num_workers=8)

    model = CustomModel()
    model.backbone.load_state_dict(torch.load(backbone_pth))
    for param in model.backbone.parameters():
        param.requires_grad = False

    callbacks = []
    callbacks += [
        ModelCheckpoint(
            monitor="val_score",
            filename="epoch-{epoch:02d}-{val_score:.3f}-{val_auc:.3f}",
            save_top_k=50,
            mode="min",
        )
    ]
    callbacks += [LearningRateMonitor(logging_interval="step")]

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        gpus=1,
        devices=[1],
        callbacks=callbacks,
        log_every_n_steps=30,
    )

    trainer.fit(model, dl_train, dl_valid)


if __name__ == "__main__":
    backbones = ["backbone.pth", "backbone.pth", "backbone.pth", "backbone.pth"]
    for fold, bb in zip(range(4), backbones):
        run(fold, bb)
