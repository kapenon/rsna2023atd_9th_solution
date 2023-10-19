from pathlib import Path

import cv2
import numpy as np
import pydicom
from tqdm import tqdm


def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    From : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype

        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)
    center = int(dcm.WindowCenter)
    width = int(dcm.WindowWidth)
    low = center - width / 2
    high = center + width / 2

    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    return pixel_array


def process(dcm):
    img = standardize_pixel_array(dcm)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    return img


def get_images(series_id, image_dir):
    file_paths = list(image_dir.glob(f"*/{series_id}/*.dcm"))

    imgs = {}
    for file_path in file_paths:
        dcm = pydicom.dcmread(file_path)
        pos_z = dcm.ImagePositionPatient[-1]  # to retrieve the order of frames
        img = process(dcm)
        imgs[pos_z] = img

    imgs = {k: v for k, v in sorted(imgs.items())}
    imgs = np.array(list(imgs.values()))
    return imgs


if __name__ == "__main__":
    root = Path(__file__).absolute().parents[1]
    data_dir = root / "data"
    out_dir = data_dir / "png_images"
    image_dir = data_dir / "train_images"

    size = 512
    it = list(image_dir.glob("*/*"))
    for s in tqdm(it, total=len(it)):
        series_id = s.stem
        imgs = get_images(series_id, image_dir)

        for i in range(len(imgs)):
            img = imgs[i]
            img = cv2.resize(img, (size, size))
            img = (img * 255).astype(np.uint8)
            out = out_dir / str(series_id) / f"{i}.png"
            out.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(out), img)
