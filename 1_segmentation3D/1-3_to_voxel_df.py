from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
image_dir = data_dir / "png_images"


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

    assert seg_pred.dtype == np.uint8

    return seg_pred


def get_voxel(seg_bin_pred, threshold=10):
    z = seg_bin_pred.sum(0).sum(0) > threshold
    zmin = np.argmax(z)
    zmax = len(z) - np.argmax(z[::-1])

    x = seg_bin_pred.sum(0).sum(1) > threshold
    xmin = np.argmax(x)
    xmax = len(x) - np.argmax(x[::-1])

    y = seg_bin_pred.sum(1).sum(1) > threshold
    ymin = np.argmax(y)
    ymax = len(y) - np.argmax(y[::-1])

    return xmin, ymin, zmin, xmax, ymax, zmax


def run():
    df = pd.read_csv(data_dir / "train_series_meta.csv")
    df["image_dir"] = str(image_dir) + "/" + df["series_id"].astype(str)
    df["seg_npy_path"] = df["series_id"].map(
        lambda sid: f"{data_dir}/seg_pred/{sid}.npy"
    )

    dfs = dict(
        series_id=[], organ=[], xmin=[], ymin=[], zmin=[], xmax=[], ymax=[], zmax=[]
    )

    targets = ["liver", "spleen", "kidney", "bowel"]
    for row in tqdm(df.itertuples(), total=len(df)):
        seg_npy = row.seg_npy_path
        seg_pred = load_seg_pred(seg_npy)
        for i in range(1, 5):
            dfs["series_id"] += [row.series_id]
            dfs["organ"] += [targets[i - 1]]

            xmin, ymin, zmin, xmax, ymax, zmax = get_voxel(seg_pred == i)
            dfs["xmin"] += [xmin]
            dfs["ymin"] += [ymin]
            dfs["zmin"] += [zmin]
            dfs["xmax"] += [xmax]
            dfs["ymax"] += [ymax]
            dfs["zmax"] += [zmax]

    df_voxel = pd.DataFrame(dfs)
    df_voxel.to_pickle(data_dir / "organ_voxel.pkl")


if __name__ == "__main__":
    run()
