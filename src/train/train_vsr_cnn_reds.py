import argparse
import random
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import pil_to_tensor
from models.vsr_cnn import VSRCNN


class VSRREDSDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        scale: int = 4,
        num_frames: int = 7,
        crop_size: int = 128,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.scale = scale
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.training = split == "train"

        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'")

        if num_frames not in {1, 3, 5, 7}:
            raise ValueError("num_frames must be one of: 1, 3, 5, 7")

        if crop_size % scale != 0:
            raise ValueError("crop_size must be divisible by scale")

        self.hr_dir = self.root / f"{split}_sharp"
        self.lr_dir = self.root / f"{split}_sharp_bicubic" / f"X{scale}"

        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR folder not found: {self.hr_dir}")

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR folder not found: {self.lr_dir}")

        self.samples = self._collect_samples()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found.\n"
                f"HR dir: {self.hr_dir}\n"
                f"LR dir: {self.lr_dir}"
            )

    def _get_sequence_dirs(self) -> list[Path]:
        sequence_dirs = sorted([p for p in self.hr_dir.iterdir() if p.is_dir()])

        if len(sequence_dirs) == 0:
            return [self.hr_dir]

        return sequence_dirs

    def _collect_samples(self) -> list[dict]:
        samples = []
        half = self.num_frames // 2

        for seq_dir in self._get_sequence_dirs():
            hr_frames = sorted(seq_dir.glob("*.png"))

            if len(hr_frames) < self.num_frames:
                continue

            for center_idx in range(half, len(hr_frames) - half):
                window_hr_paths = hr_frames[
                    center_idx - half : center_idx + half + 1
                ]

                target_hr_path = hr_frames[center_idx]
                target_rel_path = target_hr_path.relative_to(self.hr_dir)

                lr_paths = []

                missing = False

                for hr_path in window_hr_paths:
                    rel_path = hr_path.relative_to(self.hr_dir)
                    lr_path = self.lr_dir / rel_path

                    if not lr_path.exists():
                        missing = True
                        break

                    lr_paths.append(lr_path)

                if missing:
                    continue

                samples.append(
                    {
                        "lr_paths": lr_paths,
                        "hr_path": target_hr_path,
                        "name": str(target_rel_path).replace("\\", "/"),
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _random_aligned_crop(
        self,
        lr_frames: list[Image.Image],
        hr: Image.Image,
    ) -> tuple[list[Image.Image], Image.Image]:
        hr_w, hr_h = hr.size

        crop_hr = self.crop_size
        crop_lr = crop_hr // self.scale

        if hr_w < crop_hr or hr_h < crop_hr:
            return lr_frames, hr

        max_left = hr_w - crop_hr
        max_top = hr_h - crop_hr

        left_hr = random.randint(0, max_left)
        top_hr = random.randint(0, max_top)

        left_hr = left_hr - (left_hr % self.scale)
        top_hr = top_hr - (top_hr % self.scale)

        left_lr = left_hr // self.scale
        top_lr = top_hr // self.scale

        hr_crop = hr.crop(
            (
                left_hr,
                top_hr,
                left_hr + crop_hr,
                top_hr + crop_hr,
            )
        )

        lr_crops = []

        for lr in lr_frames:
            lr_crop = lr.crop(
                (
                    left_lr,
                    top_lr,
                    left_lr + crop_lr,
                    top_lr + crop_lr,
                )
            )

            lr_crops.append(lr_crop)

        return lr_crops, hr_crop

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        lr_frames = [
            Image.open(path).convert("RGB")
            for path in sample["lr_paths"]
        ]

        hr = Image.open(sample["hr_path"]).convert("RGB")

        if self.training:
            lr_frames, hr = self._random_aligned_crop(lr_frames, hr)

        lr_tensor = torch.stack(
            [pil_to_tensor(frame) for frame in lr_frames],
            dim=0,
        )

        hr_tensor = pil_to_tensor(hr)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "name": sample["name"],
        }


def sanitize_checkpoint_name(name: str) -> str:
    name = name.strip()

    if name.endswith(".pth"):
        name = name[:-4]

    name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

    if not name:
        raise ValueError("Checkpoint name cannot be empty")

    return name


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    output_path: Path,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        },
        output_path,
    )


def load_pretrained_if_needed(
    model: nn.Module,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded pretrained checkpoint from: {checkpoint_path}")


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            pred = model(lr).clamp(0.0, 1.0)
            loss = criterion(pred, hr)

            losses.append(loss.item())

    model.train()

    return sum(losses) / max(1, len(losses))


def write_config_to_tensorboard(
    writer: SummaryWriter,
    args: argparse.Namespace,
    device: torch.device,
    train_dataset_size: int,
    val_dataset_size: int,
) -> None:
    config_lines = [
        "# VSR-CNN REDS training configuration",
        "",
        f"device: {device}",
        f"train_dataset_size: {train_dataset_size}",
        f"val_dataset_size: {val_dataset_size}",
        "",
        "## Arguments",
    ]

    for key, value in vars(args).items():
        config_lines.append(f"- {key}: {value}")

    writer.add_text("config", "\n".join(config_lines), global_step=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / fine-tune VSR-CNN on REDS.")

    parser.add_argument("--root", type=str, default="data/REDS")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=7, choices=[1, 3, 5, 7])

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=8)

    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-val-samples", type=int, default=1000)

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Custom checkpoint name. If omitted, it will be requested from keyboard.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/checkpoints",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/logs/tensorboard",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional checkpoint for transfer learning / fine-tuning.",
    )

    args = parser.parse_args()

    if args.checkpoint_name is None:
        args.checkpoint_name = input("Checkpoint name: ")

    args.checkpoint_name = sanitize_checkpoint_name(args.checkpoint_name)

    checkpoint_path = Path(args.checkpoint_dir) / f"{args.checkpoint_name}.pth"
    run_log_dir = Path(args.log_dir) / args.checkpoint_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Final checkpoint will be saved to: {checkpoint_path}")
    print(f"TensorBoard logs will be saved to: {run_log_dir}")

    writer = SummaryWriter(log_dir=str(run_log_dir))

    train_dataset = VSRREDSDataset(
        root=args.root,
        split="train",
        scale=args.scale,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        max_samples=args.max_train_samples,
    )

    val_dataset = VSRREDSDataset(
        root=args.root,
        split="val",
        scale=args.scale,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = VSRCNN(
        num_frames=args.num_frames,
        scale=args.scale,
        num_channels=3,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
    ).to(device)

    load_pretrained_if_needed(
        model=model,
        checkpoint_path=args.pretrained,
        device=device,
    )

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    write_config_to_tensorboard(
        writer=writer,
        args=args,
        device=device,
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
    )

    last_train_loss = -1.0
    last_val_loss = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            pred = model(lr).clamp(0.0, 1.0)
            loss = criterion(pred, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            global_step += 1
            writer.add_scalar("Loss/train_step", loss.item(), global_step)

            progress.set_postfix(loss=f"{loss.item():.6f}")

        last_train_loss = sum(train_losses) / max(1, len(train_losses))
        last_val_loss = validate(model, val_loader, criterion, device)

        writer.add_scalar("Loss/train_epoch", last_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", last_val_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={last_train_loss:.6f}, "
            f"val_loss={last_val_loss:.6f}"
        )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        output_path=checkpoint_path,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        args=args,
    )

    writer.close()

    print("Training finished.")
    print(f"Saved final checkpoint to: {checkpoint_path}")
    print(f"TensorBoard logs saved to: {run_log_dir}")
    print(f"Final train loss: {last_train_loss:.6f}")
    print(f"Final val loss: {last_val_loss:.6f}")


if __name__ == "__main__":
    main()