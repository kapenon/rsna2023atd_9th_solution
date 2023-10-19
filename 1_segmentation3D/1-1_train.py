import gc
import os
import time
from pathlib import Path

import monai.transforms as transforms
import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torch.optim as optim
from pylab import rcParams
from seg.dataset import SEGDataset
from seg.loss import bce_dice
from seg.metrics import multilabel_dice_score
from seg.models import TimmSegModel, convert_3d
from tqdm import tqdm

rcParams["figure.figsize"] = 20, 8
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

kernel_type = (
    "timm3d_res18d_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_bs4_lr3e4_20x50ep"
)
load_kernel = None
load_last = True
n_blocks = 4
n_folds = 5
backbone = "resnet18d"

image_sizes = [128, 128, 128]  # h, w, d

init_lr = 3e-3
batch_size = 4
drop_rate = 0.0
drop_path_rate = 0.0
p_mixup = 0.1

root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
use_amp = True
num_workers = 4
out_dim = 4

n_epochs = 1000

log_dir = "./logs"
model_dir = "./models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


transforms_train = transforms.Compose(
    [
        transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        transforms.RandAffined(
            keys=["image", "mask"],
            translate_range=[int(x * y) for x, y in zip(image_sizes, [0.3, 0.3, 0.3])],
            padding_mode="zeros",
            prob=0.7,
        ),
        transforms.RandGridDistortiond(
            keys=("image", "mask"),
            prob=0.5,
            distort_limit=(-0.01, 0.01),
            mode="nearest",
        ),
    ]
)

transforms_valid = transforms.Compose([])
criterion = bce_dice


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images, gt_masks in bar:
        optimizer.zero_grad()
        images = images.cuda()
        gt_masks = gt_masks.cuda()

        with amp.autocast():
            logits = model(images)
            loss = criterion(logits, gt_masks)

        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f"smth:{np.mean(train_loss[-30:]):.4f}")

    return np.mean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    batch_metrics = [[]] * len(ths)
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, gt_masks in bar:
            images = images.cuda()
            gt_masks = gt_masks.cuda()

            logits = model(images)
            loss = criterion(logits, gt_masks)
            valid_loss.append(loss.item())
            for thi, th in enumerate(ths):
                for i in range(logits.shape[0]):
                    tmp = multilabel_dice_score(
                        y_pred=logits[i].sigmoid().cpu(),
                        y_true=gt_masks[i].cpu(),
                        threshold=th,
                    )
                    batch_metrics[thi].extend(tmp)
            bar.set_description(f"smth:{np.mean(valid_loss[-30:]):.4f}")

    metrics = [np.mean(this_metric) for this_metric in batch_metrics]
    print("best th:", ths[np.argmax(metrics)], "best dc:", np.max(metrics))

    return np.mean(valid_loss), np.max(metrics)


def run(fold):
    df = pd.read_csv("seg_train.csv")
    mask_files = []
    image_dirs = []
    for series_id in df["series_id"].values:
        mask_files += [data_dir / "segmentations" / f"{series_id}.nii"]
        image_dirs += [data_dir / "png_images" / f"{series_id}"]
    df["mask_file"] = mask_files
    df["image_dir"] = image_dirs

    log_file = os.path.join(log_dir, f"{kernel_type}.txt")
    model_file = os.path.join(model_dir, f"{kernel_type}_fold{fold}_best.pth")

    df_train = df[df["fold"] != fold].reset_index(drop=True)
    df_valid = df[df["fold"] == fold].reset_index(drop=True)
    dataset_train = SEGDataset(df_train, transforms_train, image_sizes)
    dataset_valid = SEGDataset(df_valid, transforms_valid, image_sizes)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = TimmSegModel(
        backbone,
        pretrained=True,
        drop_rate=drop_rate,
        n_blocks=n_blocks,
        drop_path_rate=drop_path_rate,
        out_dim=out_dim,
    )
    model = convert_3d(model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=init_lr)
    scaler = torch.cuda.amp.GradScaler()
    metric_best = 0.0

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, n_epochs
    )

    for epoch in range(1, n_epochs + 1):
        scheduler_cosine.step(epoch - 1)

        print(time.ctime(), "Epoch:", epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss, metric = valid_func(model, loader_valid)

        content = (
            time.ctime()
            + " "
            + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
        )
        print(content)
        with open(log_file, "a") as appender:
            appender.write(content + "\n")

        if metric > metric_best:
            print(f"metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...")
            torch.save(model.state_dict(), model_file)
            metric_best = metric

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "score_best": metric_best,
            },
            model_file.replace("_best", "_last"),
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    for fold in range(4):
        run(fold)
