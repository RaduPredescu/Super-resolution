import argparse
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

from datasets import bicubic_downsample


def read_split(root: Path, split: str, max_samples: Optional[int]) -> list[str]:
    list_file = root / f"sep_{split}list.txt"
    with list_file.open("r", encoding="utf-8") as f:
        samples = [line.strip() for line in f if line.strip()]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LR x4 frames for Vimeo-90K.")
    parser.add_argument("--root", type=str, default="data/vimeo90k")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    gt_dir = root / "GT"
    lr_dir = root / f"LR_X{args.scale}"

    samples = read_split(root, args.split, args.max_samples)

    print(f"Generating LR x{args.scale} for {len(samples)} {args.split} samples")
    print(f"Input HR: {gt_dir}")
    print(f"Output LR: {lr_dir}")

    for rel_path in tqdm(samples):
        out_dir = lr_dir / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, 8):
            hr_path = gt_dir / rel_path / f"im{i}.png"
            lr_path = out_dir / f"im{i}.png"

            if lr_path.exists():
                continue

            img = Image.open(hr_path).convert("RGB")
            lr_img = bicubic_downsample(img, args.scale)
            lr_img.save(lr_path)

    print("Done.")


if __name__ == "__main__":
    main()
