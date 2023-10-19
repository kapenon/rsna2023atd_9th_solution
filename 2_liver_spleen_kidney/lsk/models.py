import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

triple_level_targets = ["liver", "spleen", "kidney"]
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


class CustomModel(pl.LightningModule):
    def __init__(self, backbone, drop_rate_last, seq_len, in_chans, lr, n_epochs):
        self.drop_rate_last = drop_rate_last
        self.seq_len = seq_len
        self.lr = lr
        self.n_epochs = n_epochs

        out_dim = 256
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=in_chans)
        hdim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.lstm = nn.LSTM(
            hdim, out_dim, num_layers=2, bidirectional=True, batch_first=True
        )
        self.relu = nn.ReLU()

        self.conv1d_kidney = nn.Conv1d(seq_len, 1, 1)
        self.attn_kidney = Attention(feature_dim=out_dim * 2, step_dim=seq_len)
        self.attn_bn_kidney = nn.BatchNorm1d(out_dim * 2)
        self.head_kidney = nn.Sequential(
            nn.Linear(out_dim * 2 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, 3),
        )
        self.conv1d_liver = nn.Conv1d(seq_len, 1, 1)
        self.attn_liver = Attention(feature_dim=out_dim * 2, step_dim=seq_len)
        self.attn_bn_liver = nn.BatchNorm1d(out_dim * 2)
        self.head_liver = nn.Sequential(
            nn.Linear(out_dim * 2 * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, 3),
        )
        self.conv1d_spleen = nn.Conv1d(seq_len, 1, 1)
        self.attn_spleen = Attention(feature_dim=out_dim * 2, step_dim=seq_len)
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
        x, _ = self.lstm(x)

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
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=1e-5
        )  # T_max is the number of epochs
        return [optimizer], [scheduler]

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
        self.log("train_loss", loss.item())
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
