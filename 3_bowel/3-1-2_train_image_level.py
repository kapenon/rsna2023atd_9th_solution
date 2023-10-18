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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# dirs
root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
image_dir = data_dir / "png_images"


# use N * C slices per patient
D = 128
C = 4
image_sizes = (384, 384)
n_epochs = 15
p_rand_order = 0.2
lr = 8e-4
eta_min = 1e-5
bs = 32
drop_rate_last = 0
backbone = "seresnext26d_32x4d"


df_split = pd.read_csv(data_dir / "split.csv")
df_split = df_split.groupby(["patient_id"]).head(1)
df = pd.read_csv(data_dir / "bowel_image_level.csv").drop(columns=["fold"])
df = df_split.merge(df, how="left", left_on="patient_id", right_on="pid")

# select data
ppos = df[df["plabel"] == 1]
pneg = df[df["plabel"] == 0]
ppos_ipos = ppos[ppos["ilabel"] == 1]
ppos_ineg = ppos[ppos["ilabel"] == 0]

# ipos ratio = 1:10
n_ipos = len(ppos_ipos)
uses = [
    ppos_ipos,
    ppos_ineg.sample(len(ppos_ipos) * 3, random_state=42),
    pneg.sample(len(ppos_ipos) * 7, random_state=42),
]
df_use = pd.concat(uses, axis=0).reset_index(drop=True)
print(df_use.groupby("fold")["plabel", "ilabel"].mean())
print(df_use.groupby(["fold", "plabel", "ilabel"]).size())


def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    return image


transforms_train = A.ReplayCompose(
    [
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=20, border_mode=4, p=0.6
        ),
    ]
)

transforms_valid = A.ReplayCompose([])


class CLSDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index()
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = load_image(row.img_path)
        inputs = (
            self.transform(image=inputs)["image"].transpose(2, 0, 1).astype(np.float64)
        )
        inputs /= 255.0
        labels = row.ilabel

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

    def on_train_epoch_start(self):
        self.meter_train_loss = AverageMeter()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = criterion(logits, labels)
        self.meter_train_loss.update(loss.item())

        self.log("train_loss", self.meter_train_loss.avg)
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
    df_train = df_use[df_use["fold"] != fold].reset_index(drop=True)
    df_valid = df_use[df_use["fold"] == fold].reset_index(drop=True)

    ds_train = CLSDataset(df_train, transforms_train)
    ds_valid = CLSDataset(df_valid, transforms_valid)

    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=4)
    dl_valid = DataLoader(ds_valid, batch_size=bs, num_workers=4)

    model = CustomModel()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        filename="epoch-{epoch:02d}-{val_auc:.3f}",
        save_top_k=2,
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
