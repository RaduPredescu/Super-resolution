import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets import Vimeo90KDataset, tensor_to_pil, pil_to_tensor
from metrics import calculate_psnr, calculate_ssim
from models.srcnn import SRCNN


def get_checkpoint_name(checkpoint_path: str) -> str:
    """
    Extracts a clean checkpoint name from a checkpoint path.

    Example:
        results/checkpoints/srcnn_vimeo90k_20epochs.pth

    becomes:
        srcnn_vimeo90k_20epochs
    """
    name = Path(checkpoint_path).stem
    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

    if not name:
        name = "unknown_checkpoint"

    return name


def make_comparison_image(
    bicubic: Image.Image,
    srcnn: Image.Image,
    hr: Image.Image,
) -> Image.Image:
    """
    Creates a horizontal comparison image:

        Bicubic | SRCNN | HR
    """
    w, h = hr.size

    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(bicubic, (0, 0))
    canvas.paste(srcnn, (w, 0))
    canvas.paste(hr, (2 * w, 0))

    return canvas


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
    parser = argparse.ArgumentParser(description="Evaluate SRCNN on Vimeo-90K.")

    parser.add_argument("--root", type=str, default="data/vimeo90k")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SRCNN checkpoint.",
    )

    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--save-examples", type=int, default=5)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/srcnn_vimeo90k",
        help="Folder where evaluation results will be saved.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_name = get_checkpoint_name(args.checkpoint)

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples" / checkpoint_name
    examples_dir.mkdir(parents=True, exist_ok=True)

    print(f"Examples will be saved to: {examples_dir}")

    dataset = Vimeo90KDataset(
        root=args.root,
        split=args.split,
        scale=args.scale,
        max_samples=args.max_samples,
    )

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
        for idx in tqdm(range(len(dataset)), desc="Evaluating SRCNN"):
            sample = dataset[idx]
            name = sample["name"]

            hr_pil = tensor_to_pil(sample["hr"])
            w, h = hr_pil.size

            # SRCNN is frame-by-frame, so we use only the center LR frame.
            lr_center = sample["lr"][sample["lr"].shape[0] // 2]
            lr_pil = tensor_to_pil(lr_center)

            # Bicubic-upscaled LR is the input to SRCNN.
            bicubic_pil = lr_pil.resize((w, h), Image.Resampling.BICUBIC)

            x = pil_to_tensor(bicubic_pil).unsqueeze(0).to(device)

            pred = model(x).clamp(0.0, 1.0)
            pred = pred.squeeze(0).cpu()

            srcnn_np = pred.permute(1, 2, 0).numpy()
            hr_np = np.array(hr_pil).astype(np.float32) / 255.0

            psnr = calculate_psnr(srcnn_np, hr_np, crop=args.scale)
            ssim = calculate_ssim(srcnn_np, hr_np, crop=args.scale)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            rows.append(
                {
                    "sample": name,
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
                    bicubic=bicubic_pil,
                    srcnn=srcnn_img,
                    hr=hr_pil,
                )

                safe_name = name.replace("/", "_")
                comparison.save(
                    examples_dir / f"{safe_name}_bicubic_srcnn_hr.png"
                )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f"metrics_srcnn_vimeo90k_{checkpoint_name}.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "checkpoint",
                "method",
                "scale",
                "psnr",
                "ssim",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / f"summary_{checkpoint_name}.txt"

    summary_path.write_text(
        f"Dataset: Vimeo-90K\n"
        f"Split: {args.split}\n"
        f"Method: SRCNN\n"
        f"Checkpoint: {checkpoint_name}\n"
        f"Checkpoint path: {args.checkpoint}\n"
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