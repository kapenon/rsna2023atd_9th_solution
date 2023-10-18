import albumentations as A
import cv2
import pytorch_lightning as pl
import sklearn.metrics
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


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


def run(ckpt):
    trained = CustomModel.load_from_checkpoint(ckpt)
    torch.save(
        trained.backbone.state_dict(),
        "backbone.pth",
    )


if __name__ == "__main__":
    checkpoint = ""
    run(checkpoint)
