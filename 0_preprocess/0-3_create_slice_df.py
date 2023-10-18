from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm

# dirs
root = Path(__file__).absolute().parents[1]
data_dir = root / "data"
image_dir = data_dir / "png_images"

kaggle_train_images_dir = data_dir / "train_images"
png_images_dir = data_dir / "png_images"


def read_dcm(path):
    ds = pydicom.dcmread(path)
    return ds.pixel_array


def is_reversed(series_id):
    file_paths = sorted(
        kaggle_train_images_dir.glob(f"*/{series_id}/*.dcm"), key=lambda x: int(x.stem)
    )

    a = pydicom.dcmread(file_paths[0]).ImagePositionPatient[-1]
    b = pydicom.dcmread(file_paths[1]).ImagePositionPatient[-1]
    return a > b


def run():
    df = pd.read_csv(data_dir / "train_series_meta.csv")
    df_label = pd.read_csv(data_dir / "train.csv")
    df_image_level = pd.read_csv(data_dir / "image_level_labels.csv")
    df = df.merge(df_label, how="left", on="patient_id")

    df_image_level_n = pd.DataFrame(
        df_image_level.groupby("series_id").size(), columns=["count"]
    )
    df = df.merge(df_image_level_n, how="left", on="series_id")

    df["n"] = df["series_id"].map(
        lambda x: len(list((png_images_dir / str(x)).glob("*.png")))
    )
    df["ratio"] = df["count"] / df["n"]

    reversed = []
    for series_id in tqdm(df["series_id"].values):
        reversed += [is_reversed(series_id)]
    df["reversed"] = reversed

    instance_ids = []
    for row in tqdm(df.itertuples()):
        image_dir = Path(kaggle_train_images_dir / f"{row.patient_id}/{row.series_id}")
        dcm_paths = sorted(
            image_dir.glob("*.dcm"),
            key=lambda x: -int(x.stem) if row.reversed else int(x.stem),
        )
        instance_ids += [[dcm.stem for dcm in dcm_paths]]

    df["instance_id"] = instance_ids
    df["slice_id"] = df["instance_id"].map(lambda x: list(range(len(x))))

    df_extravasation = df_image_level[
        df_image_level["injury_name"] == "Active_Extravasation"
    ].reset_index(drop=True)

    series_id_to_instance_ids = (
        df_extravasation.groupby("series_id")["instance_number"].agg(list).to_dict()
    )
    df["positive_instance_id_extravasation"] = df["series_id"].map(
        series_id_to_instance_ids
    )

    res = []
    for instance_ids, series_id in tqdm(
        zip(df["instance_id"].values, df["series_id"].values)
    ):
        if series_id not in series_id_to_instance_ids.keys():
            res += [[]]
            continue

        pos_instance_ids = series_id_to_instance_ids[series_id]
        instance_id_to_slice_id = {}

        for sid, iid in enumerate(instance_ids):
            instance_id_to_slice_id[iid] = sid

        res += [[instance_id_to_slice_id[str(piid)] for piid in pos_instance_ids]]

    df["positive_slice_id_extravasation"] = res

    df_bowel = df_image_level[df_image_level["injury_name"] == "Bowel"].reset_index(
        drop=True
    )
    series_id_to_instance_ids = (
        df_bowel.groupby("series_id")["instance_number"].agg(list).to_dict()
    )
    df["positive_instance_id_bowel"] = df["series_id"].map(series_id_to_instance_ids)
    res = []
    for instance_ids, series_id in tqdm(
        zip(df["instance_id"].values, df["series_id"].values)
    ):
        if series_id not in series_id_to_instance_ids.keys():
            res += [[]]
            continue
        pos_instance_ids = series_id_to_instance_ids[series_id]
        instance_id_to_slice_id = {}

        for sid, iid in enumerate(instance_ids):
            instance_id_to_slice_id[iid] = sid

        res += [[instance_id_to_slice_id[str(piid)] for piid in pos_instance_ids]]

    df["positive_slice_id_bowel"] = res

    df.to_pickle(data_dir / "slice.pkl")


if __name__ == "__main__":
    run()
