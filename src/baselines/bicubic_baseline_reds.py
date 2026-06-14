import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from metrics import calculate_psnr, calculate_ssim


def get_reds_paths(root: Path, split: str, scale: int) -> tuple[Path, Path]:
    """
    Returns HR and LR folders for REDS.

    Expected structure:

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
    lr_upscaled: Image.Image,
    hr: Image.Image,
) -> Image.Image:
    """
    Creates comparison image:

        Bicubic | HR
    """
    w, h = hr.size

    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(lr_upscaled, (0, 0))
    canvas.paste(hr, (w, 0))

    return canvas


def safe_filename(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bicubic baseline on REDS.")

    parser.add_argument("--root", type=str, default="data/REDS")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--save-examples", type=int, default=10)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/bicubic_reds",
        help="Folder where REDS bicubic results will be saved.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Dataset: REDS")
    print(f"Split: {args.split}")
    print(f"HR dir: {hr_dir}")
    print(f"LR dir: {lr_dir}")
    print(f"Number of evaluated frames: {len(pairs)}")

    rows = []
    psnr_values = []
    ssim_values = []

    for idx, (hr_path, lr_path, rel_name) in enumerate(
        tqdm(pairs, desc="Evaluating bicubic on REDS")
    ):
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        # Upscale LR to HR size using bicubic interpolation.
        sr_img = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)

        hr_np = np.array(hr_img).astype(np.float32) / 255.0
        sr_np = np.array(sr_img).astype(np.float32) / 255.0

        psnr = calculate_psnr(sr_np, hr_np, crop=args.scale)
        ssim = calculate_ssim(sr_np, hr_np, crop=args.scale)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        rows.append(
            {
                "sample": rel_name,
                "dataset": "REDS",
                "split": args.split,
                "method": "Bicubic",
                "scale": args.scale,
                "psnr": psnr,
                "ssim": ssim,
            }
        )

        if idx < args.save_examples:
            comparison = make_comparison_image(
                lr_upscaled=sr_img,
                hr=hr_img,
            )

            comparison.save(
                examples_dir / f"{safe_filename(rel_name)}_bicubic_hr.png"
            )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    output_csv = output_dir / f"metrics_bicubic_reds_{args.split}.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "dataset",
                "split",
                "method",
                "scale",
                "psnr",
                "ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / f"summary_bicubic_reds_{args.split}.txt"

    summary_path.write_text(
        f"Dataset: REDS\n"
        f"Split: {args.split}\n"
        f"Method: Bicubic\n"
        f"Scale: x{args.scale}\n"
        f"Samples: {len(pairs)}\n"
        f"Mean PSNR: {mean_psnr:.4f}\n"
        f"Mean SSIM: {mean_ssim:.4f}\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Saved metrics to: {output_csv}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved examples to: {examples_dir}")


if __name__ == "__main__":
    main()