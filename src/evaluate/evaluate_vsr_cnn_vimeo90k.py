import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import Vimeo90KDataset, tensor_to_pil
from metrics import calculate_psnr, calculate_ssim
from models.vsr_cnn import VSRCNN


def get_checkpoint_name(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).stem
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

    if not name:
        name = "unknown_checkpoint"

    return name


def make_comparison_image(
    bicubic: Image.Image,
    vsr: Image.Image,
    hr: Image.Image,
) -> Image.Image:
    w, h = hr.size

    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(bicubic, (0, 0))
    canvas.paste(vsr, (w, 0))
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
    model: VSRCNN,
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
    parser = argparse.ArgumentParser(description="Evaluate Simple VSR-CNN on Vimeo-90K.")

    parser.add_argument("--root", type=str, default="data/vimeo90k")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=7, choices=[1, 3, 5, 7])

    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=8)

    parser.add_argument("--max-samples", type=int, default=814)
    parser.add_argument("--save-examples", type=int, default=10)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/simple_vsr_cnn_vimeo90k",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_name = get_checkpoint_name(args.checkpoint)

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples" / checkpoint_name
    examples_dir.mkdir(parents=True, exist_ok=True)

    dataset = Vimeo90KDataset(
        root=args.root,
        split="test",
        scale=args.scale,
        num_frames=args.num_frames,
        max_samples=args.max_samples,
    )

    print(f"Dataset: Vimeo-90K")
    print(f"Samples: {len(dataset)}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Examples will be saved to: {examples_dir}")

    model = VSRCNN(
        num_frames=args.num_frames,
        scale=args.scale,
        num_channels=3,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
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
        for idx in tqdm(range(len(dataset)), desc="Evaluating Simple VSR-CNN"):
            sample = dataset[idx]

            lr = sample["lr"].unsqueeze(0).to(device)
            hr = sample["hr"]
            name = sample["name"]

            pred = model(lr).clamp(0.0, 1.0)
            pred = pred.squeeze(0).cpu()

            vsr_np = pred.permute(1, 2, 0).numpy()
            hr_np = hr.permute(1, 2, 0).numpy()

            psnr = calculate_psnr(vsr_np, hr_np, crop=args.scale)
            ssim = calculate_ssim(vsr_np, hr_np, crop=args.scale)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            rows.append(
                {
                    "sample": name,
                    "dataset": "Vimeo-90K",
                    "method": "Simple VSR-CNN",
                    "checkpoint": checkpoint_name,
                    "num_frames": args.num_frames,
                    "scale": args.scale,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

            if idx < args.save_examples:
                center_lr = lr.squeeze(0)[args.num_frames // 2].cpu()
                hr_pil = tensor_to_pil(hr)
                vsr_pil = tensor_to_pil(pred)

                bicubic_pil = tensor_to_pil(center_lr).resize(
                    hr_pil.size,
                    Image.Resampling.BICUBIC,
                )

                comparison = make_comparison_image(
                    bicubic=bicubic_pil,
                    vsr=vsr_pil,
                    hr=hr_pil,
                )

                comparison.save(
                    examples_dir / f"{safe_filename(name)}_bicubic_vsr_hr.png"
                )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"metrics_simple_vsr_cnn_vimeo90k_{checkpoint_name}.csv"
    summary_path = output_dir / f"summary_simple_vsr_cnn_vimeo90k_{checkpoint_name}.txt"

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "dataset",
                "method",
                "checkpoint",
                "num_frames",
                "scale",
                "psnr",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(
        f"Dataset: Vimeo-90K\n"
        f"Method: Simple VSR-CNN\n"
        f"Checkpoint: {checkpoint_name}\n"
        f"Scale: x{args.scale}\n"
        f"Number of frames: {args.num_frames}\n"
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