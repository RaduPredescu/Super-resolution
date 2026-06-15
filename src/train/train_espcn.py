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
from models.espcn import ESPCN


class ESPCNVimeoDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        scale: int = 4,
        crop_size: int = 128,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.gt_dir = self.root / "GT"
        self.split = split
        self.scale = scale
        self.crop_size = crop_size
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

    def _load_center_frame(self, rel_path: str) -> Image.Image:
        frame_path = self.gt_dir / rel_path / "im4.png"

        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame: {frame_path}")

        return Image.open(frame_path).convert("RGB")

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        if w < self.crop_size or h < self.crop_size:
            return img

        max_left = w - self.crop_size
        max_top = h - self.crop_size

        left = random.randint(0, max_left)
        top = random.randint(0, max_top)

        left = left - (left % self.scale)
        top = top - (top % self.scale)

        return img.crop((left, top, left + self.crop_size, top + self.crop_size))

    def __getitem__(self, idx: int) -> dict:
        rel_path = self.samples[idx]

        hr = self._load_center_frame(rel_path)

        if self.training:
            hr = self._random_crop(hr)

        lr = bicubic_downsample(hr, scale=self.scale)

        return {
            "lr": pil_to_tensor(lr),
            "hr": pil_to_tensor(hr),
            "name": rel_path,
        }


class ESPCNREDSDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        scale: int = 4,
        crop_size: int = 128,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.scale = scale
        self.crop_size = crop_size
        self.training = split == "train"

        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        if crop_size % scale != 0:
            raise ValueError("crop_size must be divisible by scale")

        self.hr_dir = self.root / f"{split}_sharp"
        self.lr_dir = self.root / f"{split}_sharp_bicubic" / f"X{scale}"

        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR folder not found: {self.hr_dir}")

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR folder not found: {self.lr_dir}")

        self.samples = self._collect_pairs()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No HR/LR pairs found.\n"
                f"HR dir: {self.hr_dir}\n"
                f"LR dir: {self.lr_dir}"
            )

    def _collect_pairs(self) -> list:
        pairs = []
        hr_frames = sorted(self.hr_dir.rglob("*.png"))

        for hr_path in hr_frames:
            rel_path = hr_path.relative_to(self.hr_dir)
            lr_path = self.lr_dir / rel_path

            if not lr_path.exists():
                print(f"Warning: missing LR frame for {rel_path}")
                continue

            pairs.append(
                {
                    "hr_path": hr_path,
                    "lr_path": lr_path,
                    "name": str(rel_path).replace("\\", "/"),
                }
            )

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def _random_aligned_crop(
        self,
        hr: Image.Image,
        lr: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        hr_w, hr_h = hr.size

        crop_hr = self.crop_size
        crop_lr = crop_hr // self.scale

        if hr_w < crop_hr or hr_h < crop_hr:
            return hr, lr

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

        lr_crop = lr.crop(
            (
                left_lr,
                top_lr,
                left_lr + crop_lr,
                top_lr + crop_lr,
            )
        )

        return hr_crop, lr_crop

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        hr = Image.open(sample["hr_path"]).convert("RGB")
        lr = Image.open(sample["lr_path"]).convert("RGB")

        if self.training:
            hr, lr = self._random_aligned_crop(hr, lr)

        return {
            "lr": pil_to_tensor(lr),
            "hr": pil_to_tensor(hr),
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
        "# ESPCN training configuration",
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


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.dataset == "vimeo":
        train_dataset = ESPCNVimeoDataset(
            root=args.root,
            split="train",
            scale=args.scale,
            crop_size=args.crop_size,
            max_samples=args.max_train_samples,
        )

        val_dataset = ESPCNVimeoDataset(
            root=args.root,
            split="test",
            scale=args.scale,
            crop_size=args.crop_size,
            max_samples=args.max_val_samples,
        )

        return train_dataset, val_dataset

    if args.dataset == "reds":
        train_dataset = ESPCNREDSDataset(
            root=args.root,
            split="train",
            scale=args.scale,
            crop_size=args.crop_size,
            max_samples=args.max_train_samples,
        )

        val_dataset = ESPCNREDSDataset(
            root=args.root,
            split="val",
            scale=args.scale,
            crop_size=args.crop_size,
            max_samples=args.max_val_samples,
        )

        return train_dataset, val_dataset

    raise ValueError(f"Unknown dataset: {args.dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / fine-tune ESPCN.")

    parser.add_argument("--dataset", type=str, required=True, choices=["vimeo", "reds"])
    parser.add_argument("--root", type=str, required=True)

    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num-features", type=int, default=64)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="results/checkpoints")
    parser.add_argument("--log-dir", type=str, default="results/logs/tensorboard")

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional ESPCN checkpoint for transfer learning / fine-tuning.",
    )

    args = parser.parse_args()

    if args.checkpoint_name is None:
        args.checkpoint_name = input("Checkpoint name: ")

    args.checkpoint_name = sanitize_checkpoint_name(args.checkpoint_name)

    checkpoint_path = Path(args.checkpoint_dir) / f"{args.checkpoint_name}.pth"
    run_log_dir = Path(args.log_dir) / args.checkpoint_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Final checkpoint will be saved to: {checkpoint_path}")
    print(f"TensorBoard logs will be saved to: {run_log_dir}")

    writer = SummaryWriter(log_dir=str(run_log_dir))

    train_dataset, val_dataset = build_datasets(args)

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

    model = ESPCN(
        scale=args.scale,
        num_channels=3,
        num_features=args.num_features,
    ).to(device)

    load_pretrained_if_needed(
        model=model,
        checkpoint_path=args.pretrained,
        device=device,
    )

    criterion = nn.MSELoss()
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