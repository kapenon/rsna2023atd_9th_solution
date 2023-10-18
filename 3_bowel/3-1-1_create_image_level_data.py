import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# dirs
root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
image_dir = data_dir / "png_images"
seg_dir = data_dir / "seg_pred"
bowel_slice_dir = data_dir / "bowel_slice"
bowel_slice_dir.mkdir(exist_ok=True, parents=True)

tmp_dir = Path("./tmp")
Path(tmp_dir).mkdir(exist_ok=True, parents=True)


df = pd.read_pickle(data_dir / "slice.pkl")
df_split = pd.read_csv(data_dir / "split.csv")
df_voxel_crop = pd.read_pickle(data_dir / "organ_voxel.pkl")

low_series_ids = set(
    df.sort_values("aortic_hu").groupby("patient_id").head(1)["series_id"].values
)
df = df[df["series_id"].isin(low_series_ids)]
df = df.merge(df_split[["patient_id", "fold"]], on="patient_id").set_index("series_id")
df_voxel_crop = df_voxel_crop[
    df_voxel_crop["series_id"].isin(low_series_ids)
    & (df_voxel_crop["organ"] == "bowel")
]
df_voxel_crop = df_voxel_crop[
    ["series_id", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
].set_index("series_id")

# padding.
pad = 0.1
for c in ["xmin", "ymin"]:
    df_voxel_crop[c] = (df_voxel_crop[c] / 128.0 - pad).clip(0.0, 1.0)
for c in ["xmax", "ymax"]:
    df_voxel_crop[c] = (df_voxel_crop[c] / 128.0 + pad).clip(0.0, 1.0)


def load_image(series_id):
    paths = sorted(image_dir.glob(f"{series_id}/*.png"), key=lambda x: int(x.stem))
    # select
    slices = np.stack([cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths])
    indices = (
        np.quantile(range(len(paths)), np.linspace(0.0, 1.0, 128)).round().astype(int)
    )
    nindices = np.array([indices - 1, indices, indices + 1]).clip(0, len(slices) - 1)
    return slices[nindices].transpose(1, 2, 3, 0) / 255.0, indices  # d, h, w, c


def load_mask(series_id):
    mask = np.load(seg_dir / f"{series_id}.npy")
    mask = mask[3].transpose(2, 0, 1)
    return mask  # d, h, w


def create_series_data(series_id):
    dfs = {"sid": [], "pid": [], "plabel": [], "ilabel": [], "fold": [], "img_path": []}

    plabel = df.loc[series_id, "bowel_injury"]

    xmin, ymin, xmax, ymax = df_voxel_crop.loc[
        series_id, ["xmin", "ymin", "xmax", "ymax"]
    ].values
    zmin, zmax = df_voxel_crop.loc[series_id][["zmin", "zmax"]].values.astype(int)

    images, indices = load_image(series_id)
    mask = load_mask(series_id)
    y, x = images.shape[1], images.shape[2]
    mask = cv2.resize(mask.transpose(1, 2, 0), (x, y)).transpose(2, 0, 1)
    inp = np.concatenate([images, mask[:, :, :, np.newaxis]], axis=-1)

    pos_indices = df.loc[series_id, "positive_slice_id_bowel"]
    pid = df.loc[series_id, "patient_id"]
    label = np.isin(indices, pos_indices).astype(np.uint8)

    xmin = int(round(xmin * x))
    xmax = int(round(xmax * x))
    ymin = int(round(ymin * y))
    ymax = int(round(ymax * y))

    inp = inp[zmin:zmax, ymin:ymax, xmin:xmax]
    label = label[zmin:zmax]

    d, h, w, c = inp.shape
    inp = (
        cv2.resize(inp.transpose(1, 2, 0, 3).reshape(h, w, d * c), (384, 384))
        .reshape(384, 384, d, c)
        .transpose(2, 0, 1, 3)
    )

    inp = (inp * 255).clip(0, 255).astype(np.uint8)

    plabel = df.loc[series_id, "bowel_injury"]
    fold = df.loc[series_id, "fold"]
    series_dir = bowel_slice_dir / str(series_id)
    series_dir.mkdir(exist_ok=True, parents=True)
    for i in range(len(inp)):
        out_path = series_dir / f"{i}.png"
        cv2.imwrite(str(out_path), inp[i])
        ilabel = label[i]
        dfs["sid"] += [series_id]
        dfs["pid"] += [pid]
        dfs["plabel"] += [plabel]
        dfs["ilabel"] += [ilabel]
        dfs["fold"] += [fold]
        dfs["img_path"] += [str(out_path)]

    res_df = pd.DataFrame(dfs)
    res_df.to_csv(tmp_dir / f"{series_id}.csv", index=False)


def create_series_data_mp():
    num_processes = 32
    pool = multiprocessing.Pool(processes=num_processes)
    _ = pool.map(create_series_data, list(df.index.unique()))
    pool.close()


def run():
    create_series_data_mp()
    dfs = [pd.read_csv(csv) for csv in tmp_dir.glob("*.csv")]
    df = pd.concat(dfs, axis=0)
    df.to_csv(data_dir / "bowel_image_level.csv", index=False)


if __name__ == "__main__":
    run()
