from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Resize
from torch.utils.data import Dataset


def load_image(image_path, image_sizes):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image, (image_sizes[0], image_sizes[1]), interpolation=cv2.INTER_LINEAR
    )
    return image


def load_series_images(series_path, image_sizes):
    series_path = Path(series_path)
    t_paths = sorted(series_path.glob("*.png"), key=lambda x: int(x.stem))

    n_scans = len(t_paths)
    indices = (
        np.quantile(list(range(n_scans)), np.linspace(0.0, 1.0, image_sizes[2]))
        .round()
        .astype(int)
    )
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images += [load_image(filename, image_sizes)]
    images = np.stack(images, -1)

    if images.ndim < 4:
        images = np.expand_dims(images, 0).repeat(3, 0)  # to 3ch
    return images  # (c, h, w, d)


def load_series_mask(path):
    mask_org = nib.load(path).get_fdata()
    mask_org = mask_org.transpose(1, 0, 2)[::-1]  # h, w, d
    mask = np.zeros((4, *mask_org.shape))  # c, h, w, d

    # liver
    mask[0] = mask_org == 1
    # spleen
    mask[1] = mask_org == 2
    # kidney (merge left kidney and right kidney)
    mask[2] = np.logical_or(mask_org == 3, mask_org == 4)
    # bowel
    mask[3] = mask_org == 5

    mask = mask.astype(np.uint8) * 255
    return mask


class SEGDataset(Dataset):
    def __init__(self, df, transform, image_sizes):
        self.df = df.reset_index()
        self.tf = transform
        self.image_sizes = image_sizes
        self.tf_mask = Resize(image_sizes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = load_series_images(row.image_dir, self.image_sizes)
        mask = load_series_mask(row.mask_file)
        mask = self.tf_mask(mask).numpy()

        res = self.tf({"image": image, "mask": mask})
        image = res["image"] / 255.0
        mask = res["mask"]
        mask = (mask > 127).astype(np.float32)

        image, mask = torch.tensor(image).float(), torch.tensor(mask).float()
        return image, mask
