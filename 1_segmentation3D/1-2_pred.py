from pathlib import Path

import monai.transforms as transforms
import numpy as np
import pandas as pd
import torch
from pylab import rcParams
from seg.dataset import load_series_images
from seg.models import TimmSegModel, convert_3d
from tqdm import tqdm

rcParams["figure.figsize"] = 20, 8
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

kernel_type = (
    "timm3d_res18d_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_bs4_lr3e4_20x50ep"
)
backbone = "resnet18d"

image_sizes = [128, 128, 128]  # h, w, d


root = Path(__file__).absolute().parents[1]
data_dir = root / "data"

transforms_valid = transforms.Compose([])


def run():
    df = pd.read_csv(data_dir / "train_series_meta.csv")

    model = convert_3d(TimmSegModel(backbone, pretrained=True)).to(device)
    model.load_state_dict(torch.load(f"./models/{kernel_type}_fold0_best.pth"))

    output_dir = data_dir / "seg_pred"
    output_dir.mkdir(exist_ok=True, parents=True)
    for sid in tqdm(df["series_id"].values):
        image_dir = data_dir / "png_images" / f"{sid}"
        images = load_series_images(image_dir, image_sizes)
        images = transforms_valid({"image": images})
        images = torch.tensor(images["image"] / 255.0).float()
        images = images.view(1, *images.shape)
        with torch.no_grad():
            model.eval()
            p = model(images.cuda())
            pred = p[0].sigmoid()
            out = output_dir / f"{image_dir.name}.npy"
            np.save(out, pred.cpu().numpy())


if __name__ == "__main__":
    run()
