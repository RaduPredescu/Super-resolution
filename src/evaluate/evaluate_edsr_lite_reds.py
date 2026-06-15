import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import pil_to_tensor, tensor_to_pil
from metrics import calculate_psnr, calculate_ssim
from models.edsr_lite import EDSRLite


def get_checkpoint_name(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).stem
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

    if not name:
        name = "unknown_checkpoint"

    return name


def get_reds_paths(root: Path, split: str, scale: int) -> tuple[Path, Path]:
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
    edsr: Image.Image,
    hr: Image.Image,
) -> Image.Image:
    w, h = hr.size

    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(bicubic, (0, 0))
    canvas.paste(edsr, (w, 0))
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
    model: EDSRLite,
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
    parser = argparse.ArgumentParser(description="Evaluate EDSR-lite on REDS.")

    parser.add_argument("--root", type=str, default="data/REDS")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=8)

    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--save-examples", type=int, default=10)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/edsr_lite_reds",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_name = get_checkpoint_name(args.checkpoint)

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples" / checkpoint_name
    examples_dir.mkdir(parents=True, exist_ok=True)

    hr_dir, lr_dir = get_reds_paths(
        root=Path(args.root),
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
    print(f"Samples: {len(pairs)}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Examples will be saved to: {examples_dir}")

    model = EDSRLite(
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
        for idx, (hr_path, lr_path, rel_name) in enumerate(
            tqdm(pairs, desc="Evaluating EDSR-lite on REDS")
        ):
            hr_img = Image.open(hr_path).convert("RGB")
            lr_img = Image.open(lr_path).convert("RGB")

            lr = pil_to_tensor(lr_img).unsqueeze(0).to(device)

            pred = model(lr).clamp(0.0, 1.0)
            pred = pred.squeeze(0).cpu()

            edsr_np = pred.permute(1, 2, 0).numpy()
            hr_np = np.array(hr_img).astype(np.float32) / 255.0

            psnr = calculate_psnr(edsr_np, hr_np, crop=args.scale)
            ssim = calculate_ssim(edsr_np, hr_np, crop=args.scale)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            rows.append(
                {
                    "sample": rel_name,
                    "dataset": "REDS",
                    "split": args.split,
                    "method": "EDSR-lite",
                    "checkpoint": checkpoint_name,
                    "scale": args.scale,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

            if idx < args.save_examples:
                hr_pil = Image.open(hr_path).convert("RGB")
                edsr_pil = tensor_to_pil(pred)

                bicubic_pil = lr_img.resize(
                    hr_pil.size,
                    Image.Resampling.BICUBIC,
                )

                comparison = make_comparison_image(
                    bicubic=bicubic_pil,
                    edsr=edsr_pil,
                    hr=hr_pil,
                )

                comparison.save(
                    examples_dir / f"{safe_filename(rel_name)}_bicubic_edsr_hr.png"
                )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"metrics_edsr_lite_reds_{args.split}_{checkpoint_name}.csv"
    summary_path = output_dir / f"summary_edsr_lite_reds_{args.split}_{checkpoint_name}.txt"

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
        f"Dataset: REDS\n"
        f"Split: {args.split}\n"
        f"Method: EDSR-lite\n"
        f"Checkpoint: {checkpoint_name}\n"
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