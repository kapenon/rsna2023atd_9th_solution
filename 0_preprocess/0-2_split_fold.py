from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

root = Path(__file__).absolute().parents[1]
data_dir = root / "data"

k = 4

cols = [
    "bowel_injury",
    "extravasation_injury",
    "kidney_low",
    "kidney_high",
    "liver_low",
    "liver_high",
    "spleen_low",
    "spleen_high",
]


def run():
    df = pd.read_csv(data_dir / "train.csv")

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    splits = mskf.split(df, y=df[cols])

    df["fold"] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df.to_csv(data_dir / "split.csv", index=False)

    for col in cols:
        print(df.groupby(["fold", col]).size())


if __name__ == "__main__":
    run()
