from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import bicubic_downsample, pil_to_tensor, tensor_to_pil
from metrics import calculate_psnr, calculate_ssim
from models.fsrcnn import FSRCNN


class FSRCNNVimeoEvalDataset:
    def __init__(
        self,
        root: str,
        scale: int = 4,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.gt_dir = self.root / "GT"
        self.scale = scale

        list_file = self.root / "sep_testlist.txt"

        if not list_file.exists():
            raise FileNotFoundError(f"Missing split file: {list_file}")

        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Missing GT folder: {self.gt_dir}")

        with list_file.open("r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f if line.strip()]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {list_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rel_path = self.samples[idx]
        hr_path = self.gt_dir / rel_path / "im4.png"

        if not hr_path.exists():
            raise FileNotFoundError(f"Missing frame: {hr_path}")

        hr = Image.open(hr_path).convert("RGB")
        lr = bicubic_downsample(hr, scale=self.scale)

        return {
            "lr": pil_to_tensor(lr),
            "hr": pil_to_tensor(hr),
            "name": rel_path,
        }


class FSRCNNREDSEvalDataset:
    def __init__(
        self,
        root: str,
        split: str = "val",
        scale: int = 4,
        max_samples: int | None = 1000,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.scale = scale

        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.hr_dir = self.root / f"{split}_sharp"
        self.lr_dir = self.root / f"{split}_sharp_bicubic" / f"X{scale}"

        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR folder not found: {self.hr_dir}")

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR folder not found: {self.lr_dir}")

        self.samples = self._collect_pairs()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No HR/LR pairs found.\n"
                f"HR dir: {self.hr_dir}\n"
                f"LR dir: {self.lr_dir}"
            )

    def _collect_pairs(self) -> list:
        pairs = []
        hr_frames = sorted(self.hr_dir.rglob("*.png"))

        for hr_path in hr_frames:
            rel_path = hr_path.relative_to(self.hr_dir)
            lr_path = self.lr_dir / rel_path

            if not lr_path.exists():
                print(f"Warning: missing LR frame for {rel_path}")
                continue

            pairs.append(
                {
                    "hr_path": hr_path,
                    "lr_path": lr_path,
                    "name": str(rel_path).replace("\\", "/"),
                }
            )

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        hr = Image.open(sample["hr_path"]).convert("RGB")
        lr = Image.open(sample["lr_path"]).convert("RGB")

        return {
            "lr": pil_to_tensor(lr),
            "hr": pil_to_tensor(hr),
            "name": sample["name"],
        }


def get_checkpoint_name(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).stem
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

    if not name:
        name = "unknown_checkpoint"

    return name


def make_comparison_image(
    bicubic: Image.Image,
    fsrcnn: Image.Image,
    hr: Image.Image,
) -> Image.Image:
    w, h = hr.size

    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(bicubic, (0, 0))
    canvas.paste(fsrcnn, (w, 0))
    canvas.paste(hr, (2 * w, 0))

    return canvas


def safe_filename(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def load_checkpoint(
    model: FSRCNN,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from: {checkpoint_path}")


def build_dataset(args: argparse.Namespace):
    if args.dataset == "vimeo":
        return FSRCNNVimeoEvalDataset(
            root=args.root,
            scale=args.scale,
            max_samples=args.max_samples,
        )

    if args.dataset == "reds":
        return FSRCNNREDSEvalDataset(
            root=args.root,
            split=args.split,
            scale=args.scale,
            max_samples=args.max_samples,
        )

    raise ValueError(f"Unknown dataset: {args.dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FSRCNN.")

    parser.add_argument("--dataset", type=str, required=True, choices=["vimeo", "reds"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument("--d", type=int, default=56)
    parser.add_argument("--s", type=int, default=12)
    parser.add_argument("--m", type=int, default=4)

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=10)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fsrcnn",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_name = get_checkpoint_name(args.checkpoint)

    output_dir = Path(args.output_dir) / args.dataset
    examples_dir = output_dir / "examples" / checkpoint_name
    examples_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(dataset)}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Examples will be saved to: {examples_dir}")

    model = FSRCNN(
        scale=args.scale,
        num_channels=3,
        d=args.d,
        s=args.s,
        m=args.m,
    ).to(device)

    load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.eval()

    rows = []
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating FSRCNN"):
            sample = dataset[idx]

            lr = sample["lr"].unsqueeze(0).to(device)
            hr = sample["hr"]
            name = sample["name"]

            pred = model(lr).clamp(0.0, 1.0)
            pred = pred.squeeze(0).cpu()

            fsrcnn_np = pred.permute(1, 2, 0).numpy()
            hr_np = hr.permute(1, 2, 0).numpy()

            psnr = calculate_psnr(fsrcnn_np, hr_np, crop=args.scale)
            ssim = calculate_ssim(fsrcnn_np, hr_np, crop=args.scale)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            rows.append(
                {
                    "sample": name,
                    "dataset": args.dataset,
                    "split": args.split if args.dataset == "reds" else "test",
                    "method": "FSRCNN",
                    "checkpoint": checkpoint_name,
                    "scale": args.scale,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

            if idx < args.save_examples:
                hr_pil = tensor_to_pil(hr)
                fsrcnn_pil = tensor_to_pil(pred)

                center_lr_pil = tensor_to_pil(sample["lr"])
                bicubic_pil = center_lr_pil.resize(
                    hr_pil.size,
                    Image.Resampling.BICUBIC,
                )

                comparison = make_comparison_image(
                    bicubic=bicubic_pil,
                    fsrcnn=fsrcnn_pil,
                    hr=hr_pil,
                )

                comparison.save(
                    examples_dir / f"{safe_filename(name)}_bicubic_fsrcnn_hr.png"
                )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"metrics_fsrcnn_{args.dataset}_{checkpoint_name}.csv"
    summary_path = output_dir / f"summary_fsrcnn_{args.dataset}_{checkpoint_name}.txt"

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "dataset",
                "split",
                "method",
                "checkpoint",
                "scale",
                "psnr",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(
        f"Dataset: {args.dataset}\n"
        f"Method: FSRCNN\n"
        f"Checkpoint: {checkpoint_name}\n"
        f"Scale: x{args.scale}\n"
        f"Samples: {len(dataset)}\n"
        f"Mean PSNR: {mean_psnr:.4f}\n"
        f"Mean SSIM: {mean_ssim:.4f}\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Saved metrics to: {output_csv}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved examples to: {examples_dir}")


if __name__ == "__main__":
    main()