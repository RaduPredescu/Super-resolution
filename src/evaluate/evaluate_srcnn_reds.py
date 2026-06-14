import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import pil_to_tensor
from metrics import calculate_psnr, calculate_ssim
from models.srcnn import SRCNN


def get_checkpoint_name(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).stem
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

    if not name:
        name = "unknown_checkpoint"

    return name


def get_reds_paths(root: Path, split: str, scale: int) -> tuple[Path, Path]:
    """
    Expected REDS structure:

        data/REDS/
        ├── train_sharp/
        ├── train_sharp_bicubic/
        │   └── X4/
        ├── val_sharp/
        └── val_sharp_bicubic/
            └── X4/
    """

    hr_dir = root / f"{split}_sharp"
    lr_dir = root / f"{split}_sharp_bicubic" / f"X{scale}"

    if not hr_dir.exists():
        raise FileNotFoundError(f"HR folder not found: {hr_dir}")

    if not lr_dir.exists():
        raise FileNotFoundError(f"LR folder not found: {lr_dir}")

    return hr_dir, lr_dir


def collect_frame_pairs(
    hr_dir: Path,
    lr_dir: Path,
    max_samples: int | None = None,
) -> list[tuple[Path, Path, str]]:
    """
    Collects matching HR/LR frame pairs.

    Example:
        HR: data/REDS/val_sharp/000/00000000.png
        LR: data/REDS/val_sharp_bicubic/X4/000/00000000.png
    """

    hr_frames = sorted(hr_dir.rglob("*.png"))

    pairs = []

    for hr_path in hr_frames:
        rel_path = hr_path.relative_to(hr_dir)
        lr_path = lr_dir / rel_path

        if not lr_path.exists():
            print(f"Warning: missing LR frame for {rel_path}")
            continue

        pairs.append((hr_path, lr_path, str(rel_path).replace("\\", "/")))

        if max_samples is not None and len(pairs) >= max_samples:
            break

    if not pairs:
        raise RuntimeError(
            f"No HR/LR pairs found.\n"
            f"HR dir: {hr_dir}\n"
            f"LR dir: {lr_dir}"
        )

    return pairs


def make_comparison_image(
    bicubic: Image.Image,
    srcnn: Image.Image,
    hr: Image.Image,
) -> Image.Image:
    """
    Creates comparison image:

        Bicubic | SRCNN | HR
    """

    w, h = hr.size

    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(bicubic, (0, 0))
    canvas.paste(srcnn, (w, 0))
    canvas.paste(hr, (2 * w, 0))

    return canvas


def safe_filename(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def load_srcnn_checkpoint(
    model: SRCNN,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SRCNN on REDS.")

    parser.add_argument("--root", type=str, default="data/REDS")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SRCNN checkpoint.",
    )

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--save-examples", type=int, default=10)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/srcnn_reds",
        help="Folder where SRCNN REDS evaluation results will be saved.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_name = get_checkpoint_name(args.checkpoint)

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    examples_dir = output_dir / "examples" / checkpoint_name
    examples_dir.mkdir(parents=True, exist_ok=True)

    print(f"Examples will be saved to: {examples_dir}")

    hr_dir, lr_dir = get_reds_paths(
        root=root,
        split=args.split,
        scale=args.scale,
    )

    pairs = collect_frame_pairs(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        max_samples=args.max_samples,
    )

    print("Dataset: REDS")
    print(f"Split: {args.split}")
    print(f"HR dir: {hr_dir}")
    print(f"LR dir: {lr_dir}")
    print(f"Number of evaluated frames: {len(pairs)}")

    model = SRCNN(num_channels=3).to(device)

    load_srcnn_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.eval()

    rows = []
    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for idx, (hr_path, lr_path, rel_name) in enumerate(
            tqdm(pairs, desc="Evaluating SRCNN on REDS")
        ):
            hr_img = Image.open(hr_path).convert("RGB")
            lr_img = Image.open(lr_path).convert("RGB")

            # SRCNN expects bicubic-upscaled LR as input.
            bicubic_img = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)

            x = pil_to_tensor(bicubic_img).unsqueeze(0).to(device)

            pred = model(x).clamp(0.0, 1.0)
            pred = pred.squeeze(0).cpu()

            srcnn_np = pred.permute(1, 2, 0).numpy()
            hr_np = np.array(hr_img).astype(np.float32) / 255.0

            psnr = calculate_psnr(srcnn_np, hr_np, crop=args.scale)
            ssim = calculate_ssim(srcnn_np, hr_np, crop=args.scale)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            rows.append(
                {
                    "sample": rel_name,
                    "dataset": "REDS",
                    "split": args.split,
                    "checkpoint": checkpoint_name,
                    "method": "SRCNN",
                    "scale": args.scale,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

            if idx < args.save_examples:
                srcnn_img = Image.fromarray(
                    (srcnn_np * 255.0)
                    .round()
                    .clip(0, 255)
                    .astype(np.uint8)
                )

                comparison = make_comparison_image(
                    bicubic=bicubic_img,
                    srcnn=srcnn_img,
                    hr=hr_img,
                )

                comparison.save(
                    examples_dir / f"{safe_filename(rel_name)}_bicubic_srcnn_hr.png"
                )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"metrics_srcnn_reds_{args.split}_{checkpoint_name}.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "dataset",
                "split",
                "checkpoint",
                "method",
                "scale",
                "psnr",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / f"summary_srcnn_reds_{args.split}_{checkpoint_name}.txt"

    summary_path.write_text(
        f"Dataset: REDS\n"
        f"Split: {args.split}\n"
        f"Method: SRCNN\n"
        f"Checkpoint: {checkpoint_name}\n"
        f"Checkpoint path: {args.checkpoint}\n"
        f"Scale: x{args.scale}\n"
        f"Samples: {len(pairs)}\n"
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