import argparse
from pathlib import Path
from PIL import Image


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Vimeo-90K folder structure.")
    parser.add_argument("--root", type=str, default="data/vimeo90k")
    args = parser.parse_args()

    root = Path(args.root)
    gt_dir = root / "GT"
    train_list = root / "sep_trainlist.txt"
    test_list = root / "sep_testlist.txt"

    print(f"Checking: {root.resolve()}")

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT folder not found: {gt_dir}")
    if not train_list.exists():
        raise FileNotFoundError(f"Train list not found: {train_list}")
    if not test_list.exists():
        raise FileNotFoundError(f"Test list not found: {test_list}")

    print(f"Train samples: {count_lines(train_list)}")
    print(f"Test samples:  {count_lines(test_list)}")

    with train_list.open("r", encoding="utf-8") as f:
        first_sample = next(line.strip() for line in f if line.strip())

    sample_dir = gt_dir / first_sample
    print(f"First train sample: {first_sample}")
    print(f"Sample folder: {sample_dir}")

    if not sample_dir.exists():
        raise FileNotFoundError(f"First sample folder not found: {sample_dir}")

    for i in range(1, 8):
        frame_path = sample_dir / f"im{i}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing expected frame: {frame_path}")

    img = Image.open(sample_dir / "im4.png").convert("RGB")
    print(f"Center frame size: {img.size[0]} x {img.size[1]}")
    print("Vimeo-90K structure looks OK.")


if __name__ == "__main__":
    main()