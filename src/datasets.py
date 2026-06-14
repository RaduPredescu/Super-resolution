from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor_to_numpy_img(tensor)
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def bicubic_downsample(img: Image.Image, scale: int = 4) -> Image.Image:
    w, h = img.size
    new_w, new_h = w // scale, h // scale

    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"Image too small for scale x{scale}: {img.size}")

    return img.resize((new_w, new_h), Image.Resampling.BICUBIC)


class Vimeo90KDataset(Dataset):
    """
    PyTorch Dataset for Vimeo-90K septuplet.

    Expected structure:

        data/vimeo90k/
        ├── GT/
        ├── sep_trainlist.txt
        └── sep_testlist.txt

    Each sample returns:
        lr: T x C x h x w
        hr: C x H x W
        name: sequence name
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        scale: int = 4,
        num_frames: int = 7,
        target_idx: Optional[int] = None,
        max_samples: Optional[int] = None,
        lr_dir: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.gt_dir = self.root / "GT"
        self.scale = scale
        self.num_frames = num_frames
        self.target_idx = target_idx if target_idx is not None else num_frames // 2

        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        list_file = self.root / f"sep_{split}list.txt"

        if not list_file.exists():
            raise FileNotFoundError(f"Missing split file: {list_file}")

        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Missing GT folder: {self.gt_dir}")

        with list_file.open("r", encoding="utf-8") as f:
            self.samples: List[str] = [line.strip() for line in f if line.strip()]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.lr_dir = Path(lr_dir) if lr_dir is not None else None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_gt_frame(self, rel_path: str, frame_idx: int) -> Image.Image:
        frame_path = self.gt_dir / rel_path / f"im{frame_idx}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame: {frame_path}")
        return Image.open(frame_path).convert("RGB")

    def _load_or_generate_lr_frame(self, rel_path: str, frame_idx: int, hr_img: Image.Image) -> Image.Image:
        if self.lr_dir is not None:
            lr_path = self.lr_dir / rel_path / f"im{frame_idx}.png"
            if lr_path.exists():
                return Image.open(lr_path).convert("RGB")

        return bicubic_downsample(hr_img, self.scale)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel_path = self.samples[idx]

        hr_frames: List[Image.Image] = []
        lr_frames: List[Image.Image] = []

        for frame_idx in range(1, self.num_frames + 1):
            hr_img = self._load_gt_frame(rel_path, frame_idx)
            lr_img = self._load_or_generate_lr_frame(rel_path, frame_idx, hr_img)

            hr_frames.append(hr_img)
            lr_frames.append(lr_img)

        target_hr = hr_frames[self.target_idx]

        lr_tensor = torch.stack([pil_to_tensor(img) for img in lr_frames], dim=0)
        hr_tensor = pil_to_tensor(target_hr)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "name": rel_path,
        }
