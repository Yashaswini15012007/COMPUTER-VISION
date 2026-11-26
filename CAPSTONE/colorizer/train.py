import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import CHECKPOINT_BEST, CHECKPOINT_DIR, CHECKPOINT_LATEST


class LabDataset(Dataset):
    """Loads preprocessed .pt tensors containing L, a, and b channels."""

    def __init__(self, root_dir: str):
        self.root = Path(root_dir).expanduser().resolve()
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found in {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = torch.load(self.files[idx], map_location="cpu")
        return tensor[0:1], tensor[1:]


def train(args):
    if len(args.dataset) < 2:
        raise RuntimeError("At least two processed samples are required for training and validation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.unet import UNet

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_scheduler
        else None
    )
    scaler = GradScaler(enabled=args.mixed_precision and device.type == "cuda")
    criterion = nn.L1Loss()

    val_size = max(1, int(len(args.dataset) * args.val_split))
    train_size = len(args.dataset) - val_size
    train_dataset, val_dataset = random_split(
        args.dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    writer = SummaryWriter(log_dir=args.log_dir)
    best_loss = float("inf")
    global_step = 0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for L, ab in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
                L = L.to(device, non_blocking=True)
                ab = ab.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=scaler.is_enabled()):
                    loss = criterion(model(L), ab)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                global_step += 1

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for L, ab in val_loader:
                    L = L.to(device, non_blocking=True)
                    ab = ab.to(device, non_blocking=True)
                    val_loss += criterion(model(L), ab).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
            writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch)

            checkpoint = {
                "epoch": epoch,
                "validation_loss": avg_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(checkpoint, CHECKPOINT_LATEST)
            print(f"Saved latest checkpoint: {CHECKPOINT_LATEST}")
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(checkpoint, CHECKPOINT_BEST)
                print(f"Saved best checkpoint: {CHECKPOINT_BEST} (validation loss {best_loss:.6f})")
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, validation_loss={avg_val_loss:.6f}")

            if scheduler is not None:
                scheduler.step()
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lab colorization UNet")
    parser.add_argument("--train-data", type=str, default=str(Path(__file__).parent / "data" / "processed"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--log-dir", type=str, default=str(Path(__file__).parent / "logs"))
    parser.add_argument("--lr-scheduler", action="store_true", help="Use cosine annealing LR scheduler")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.val_split < 1:
        parser.error("--val-split must be between 0 and 1.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    args.dataset = LabDataset(args.train_data)
    train(args)
