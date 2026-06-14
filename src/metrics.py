import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    if img.ndim == 3:
        return img[border:-border, border:-border, :]
    return img[border:-border, border:-border]


def calculate_psnr(sr: np.ndarray, hr: np.ndarray, crop: int = 0) -> float:
    sr = crop_border(sr, crop)
    hr = crop_border(hr, crop)
    sr = np.clip(sr, 0.0, 1.0)
    hr = np.clip(hr, 0.0, 1.0)
    return float(peak_signal_noise_ratio(hr, sr, data_range=1.0))


def calculate_ssim(sr: np.ndarray, hr: np.ndarray, crop: int = 0) -> float:
    sr = crop_border(sr, crop)
    hr = crop_border(hr, crop)
    sr = np.clip(sr, 0.0, 1.0)
    hr = np.clip(hr, 0.0, 1.0)
    return float(structural_similarity(hr, sr, channel_axis=2, data_range=1.0))
