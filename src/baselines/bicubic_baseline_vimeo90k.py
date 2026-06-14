import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import Vimeo90KDataset, tensor_to_numpy_img, tensor_to_pil
from metrics import calculate_psnr, calculate_ssim


def make_comparison_image(sr: Image.Image, hr: Image.Image) -> Image.Image:
    w, h = hr.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(sr, (0, 0))
    canvas.paste(hr, (w, 0))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Bicubic baseline on Vimeo-90K.")
    parser.add_argument("--root", type=str, default="data/vimeo90k")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--save-examples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/bicubic_vimeo90k")
    parser.add_argument("--use-saved-lr", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    lr_dir = None
    if args.use_saved_lr:
        candidate = Path(args.root) / f"LR_X{args.scale}"
        if candidate.exists():
            lr_dir = str(candidate)
            print(f"Using saved LR frames from: {candidate}")
        else:
            print(f"Saved LR folder not found: {candidate}. Generating LR on the fly.")

    dataset = Vimeo90KDataset(
        root=args.root,
        split=args.split,
        scale=args.scale,
        max_samples=args.max_samples,
        lr_dir=lr_dir,
    )

    rows = []
    psnr_values = []
    ssim_values = []

    print(f"Evaluating bicubic baseline on {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        name = sample["name"]

        lr_center = sample["lr"][sample["lr"].shape[0] // 2]
        hr_tensor = sample["hr"]

        lr_pil = tensor_to_pil(lr_center)
        hr_pil = tensor_to_pil(hr_tensor)

        sr_pil = lr_pil.resize(hr_pil.size, Image.Resampling.BICUBIC)

        sr_np = np.array(sr_pil).astype(np.float32) / 255.0
        hr_np = tensor_to_numpy_img(hr_tensor)

        psnr = calculate_psnr(sr_np, hr_np, crop=args.scale)
        ssim = calculate_ssim(sr_np, hr_np, crop=args.scale)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

        rows.append({
            "sample": name,
            "method": "bicubic",
            "scale": args.scale,
            "psnr": psnr,
            "ssim": ssim,
        })

        if idx < args.save_examples:
            comparison = make_comparison_image(sr_pil, hr_pil)
            safe_name = name.replace("/", "_")
            comparison.save(examples_dir / f"{safe_name}_bicubic_vs_hr.png")

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    output_csv = output_dir / "metrics_bicubic_vimeo90k.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "method", "scale", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        f"Dataset: Vimeo-90K\n"
        f"Split: {args.split}\n"
        f"Method: Bicubic\n"
        f"Scale: x{args.scale}\n"
        f"Samples: {len(dataset)}\n"
        f"Mean PSNR: {mean_psnr:.4f}\n"
        f"Mean SSIM: {mean_ssim:.4f}\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Saved metrics to: {output_csv}")
    print(f"Saved examples to: {examples_dir}")


if __name__ == "__main__":
    main()
