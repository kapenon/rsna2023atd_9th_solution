from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from lsk.dataset import CLSDataset
from lsk.models import CustomModel
from lsk.utils import set_seed
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

set_seed()

# dirs
root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
image_dir = data_dir / "png_images"

# params
C = 3
image_sizes = [96, 256, 256]  # d, h, wi
n_epochs = 15
p_rand_order = 0.2
lr = 1e-4
bs = 4
drop_rate_last = 0
backbone = "seresnext26d_32x4d"


def seg_slice_ids(series_path):
    series_path = Path(series_path)
    n_slices = len(list(series_path.glob("*.png")))
    indices = (
        np.quantile(range(n_slices), np.linspace(0.0, 1.0, 128)).round().astype(int)
    )
    return indices, n_slices


def prepare_df():
    df_split = pd.read_csv(data_dir / "split.csv")
    df = pd.read_csv(data_dir / "train_series_meta.csv")
    df = df.merge(df_split, how="left", on="patient_id")
    df["image_dir"] = str(image_dir) + "/" + df["series_id"].astype(str)

    # Keep only lower HU.
    df = (
        df.sort_values(["aortic_hu"])
        .reset_index(drop=True)
        .groupby("patient_id")
        .head(1)
        .reset_index(drop=True)
    )
    df_voxel = pd.read_pickle(data_dir / "organ_voxel.pkl")
    df = df.merge(df_voxel, on="series_id")

    # Attach seg pred info.
    seg_slice_ids_list = []
    n_slices_list = []
    for sid in df["series_id"].values:
        indices, n_slices = seg_slice_ids(image_dir / str(sid))
        seg_slice_ids_list += [indices]
        n_slices_list += [n_slices]
    df["seg_slice_ids"] = seg_slice_ids_list
    df["n_slices"] = n_slices_list
    df["seg_npy_path"] = df["series_id"].map(
        lambda sid: data_dir / "seg_pred" / f"{sid}.npy"
    )

    # Convert to int label.
    binary_targets = ["bowel", "extravasation"]
    triple_level_targets = ["liver", "spleen", "kidney"]
    for t in binary_targets:
        df[t] = df[f"{t}_injury"]
    for t in triple_level_targets:
        df[t] = np.argmax(df[[f"{t}_healthy", f"{t}_low", f"{t}_high"]].values, axis=-1)

    # Merge voxels and add enlarge voxels.
    df_triple_organs = df[df["organ"].isin(triple_level_targets)]

    pad = 0.01
    df_min = df_triple_organs.groupby("series_id")[["xmin", "ymin", "zmin"]].agg(min)
    df_min[["xmin", "ymin"]] = (df_min[["xmin", "ymin"]] / 128.0 - pad).clip(0.0, 1.0)
    df_max = df_triple_organs.groupby("series_id")[["xmax", "ymax", "zmax"]].agg(max)
    df_max[["xmax", "ymax"]] = (df_max[["xmax", "ymax"]] / 128.0 + pad).clip(0.0, 1.0)

    df = (
        df.groupby("series_id")
        .head(1)
        .drop(columns=["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"])
    )
    df = df.merge(df_min, how="left", on="series_id")
    df = df.merge(df_max, how="left", on="series_id")
    return df


# bbox: 0-1 np.array (xmin, ymin, xmax, ymax)
transforms_train = A.ReplayCompose(
    [
        A.Resize(image_sizes[1], image_sizes[2]),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=4, p=0.8
        ),
    ]
)
transforms_valid = A.ReplayCompose(
    [
        A.Resize(image_sizes[1], image_sizes[2]),
    ]
)


def run(fold):
    df = prepare_df()
    df_train = df[df["fold"] != fold].reset_index(drop=True)
    df_valid = df[df["fold"] == fold].reset_index(drop=True)

    ds_train = CLSDataset(
        df_train, transforms_train, p_rand_order, n_channels=3, image_sizes=image_sizes
    )
    ds_valid = CLSDataset(
        df_valid, transforms_valid, p_rand_order, n_channels=3, image_sizes=image_sizes
    )

    dl_train = DataLoader(
        ds_train, batch_size=bs, shuffle=True, drop_last=True, num_workers=4
    )
    dl_valid = DataLoader(ds_valid, batch_size=bs, num_workers=4)

    model = CustomModel(
        backbone=backbone,
        drop_rate_last=drop_rate_last,
        seq_len=image_sizes[0],
        in_chans=4,
        lr=lr,
        n_epochs=n_epochs,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_score",
            filename="best_model",
            save_top_k=1,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        gpus=1,
        devices=[1],
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(model, dl_train, dl_valid)


if __name__ == "__main__":
    for fold in range(4):
        run(fold)
