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

from datasets import bicubic_downsample, pil_to_tensor
from models.vsr_cnn import VSRCNN


def get_centered_frame_indices(num_frames: int) -> list[int]:
    """
    Vimeo-90K septuplet has frames im1.png ... im7.png.
    Target is always the center frame im4.png.

    num_frames=7 -> [1,2,3,4,5,6,7]
    num_frames=5 -> [2,3,4,5,6]
    num_frames=3 -> [3,4,5]
    num_frames=1 -> [4]
    """

    if num_frames not in {1, 3, 5, 7}:
        raise ValueError("num_frames must be one of: 1, 3, 5, 7")

    center = 4
    half = num_frames // 2

    return list(range(center - half, center + half + 1))


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()

class SimpleVSRVimeoDataset(Dataset):
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
        self.gt_dir = self.root / "GT"
        self.split = split
        self.scale = scale
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.frame_indices = get_centered_frame_indices(num_frames)
        self.training = split == "train"

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        if crop_size % scale != 0:
            raise ValueError("crop_size must be divisible by scale")

        list_file = self.root / f"sep_{split}list.txt"

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

    def _load_frame(self, rel_path: str, frame_idx: int) -> Image.Image:
        frame_path = self.gt_dir / rel_path / f"im{frame_idx}.png"

        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame: {frame_path}")

        return Image.open(frame_path).convert("RGB")

    def _random_crop_all_frames(
        self,
        frames: list[Image.Image],
    ) -> list[Image.Image]:
        w, h = frames[0].size

        if w < self.crop_size or h < self.crop_size:
            return frames

        max_left = w - self.crop_size
        max_top = h - self.crop_size

        left = random.randint(0, max_left)
        top = random.randint(0, max_top)

        left = left - (left % self.scale)
        top = top - (top % self.scale)

        cropped = []

        for frame in frames:
            cropped.append(
                frame.crop(
                    (
                        left,
                        top,
                        left + self.crop_size,
                        top + self.crop_size,
                    )
                )
            )

        return cropped

    def __getitem__(self, idx: int) -> dict:
        rel_path = self.samples[idx]

        hr_frames = [
            self._load_frame(rel_path, frame_idx)
            for frame_idx in self.frame_indices
        ]

        target_hr = self._load_frame(rel_path, 4)

        if self.training:
            # Crop all input frames plus target frame at the same location.
            all_frames = hr_frames + [target_hr]
            all_frames = self._random_crop_all_frames(all_frames)
            hr_frames = all_frames[:-1]
            target_hr = all_frames[-1]

        lr_frames = [
            bicubic_downsample(frame, scale=self.scale)
            for frame in hr_frames
        ]

        lr_tensor = torch.stack(
            [pil_to_tensor(frame) for frame in lr_frames],
            dim=0,
        )

        hr_tensor = pil_to_tensor(target_hr)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "name": rel_path,
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
        "# Simple VSR-CNN Vimeo-90K training configuration",
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
    parser = argparse.ArgumentParser(description="Train Simple VSR-CNN on Vimeo-90K.")

    parser.add_argument("--root", type=str, default="data/vimeo90k")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=7, choices=[1, 3, 5, 7])

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=8)

    parser.add_argument("--max-train-samples", type=int, default=6206)
    parser.add_argument("--max-val-samples", type=int, default=814)

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

    train_dataset = SimpleVSRVimeoDataset(
        root=args.root,
        split="train",
        scale=args.scale,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        max_samples=args.max_train_samples,
    )

    val_dataset = SimpleVSRVimeoDataset(
        root=args.root,
        split="test",
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

    criterion = CharbonnierLoss()
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