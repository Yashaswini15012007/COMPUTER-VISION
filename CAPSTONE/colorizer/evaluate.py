import argparse
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from inference import load_model
from utils import psnr, ssim, lpips_metric, lab_tensor_to_rgb
from utils import get_checkpoint_path
from train import LabDataset

def save_comparison_grid(L, ab_gt, ab_pred, save_path, num_images=8):
    """Save a side‑by‑side grid of grayscale, ground‑truth colour, and predicted colour.
    L: (B, 1, H, W) tensor
    ab_gt: (B, 2, H, W) tensor
    ab_pred: (B, 2, H, W) tensor
    """
    L = L.cpu()
    ab_gt = ab_gt.cpu()
    ab_pred = ab_pred.cpu()
    B = min(num_images, L.size(0))
    rows = 3  # grayscale, ground truth, prediction
    cols = B
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i in range(B):
        # Grayscale image (L channel expanded to RGB)
        L_img = (L[i, 0].numpy() * 255).astype(np.uint8)
        L_rgb = np.stack([L_img] * 3, axis=-1)
        # Ground truth colour
        gt_lab = torch.cat([L[i], ab_gt[i]], dim=0)
        gt_rgb = lab_tensor_to_rgb(gt_lab.unsqueeze(0))[0]
        # Prediction colour
        pred_lab = torch.cat([L[i], ab_pred[i]], dim=0)
        pred_rgb = lab_tensor_to_rgb(pred_lab.unsqueeze(0))[0]
        axes[0, i].imshow(L_rgb)
        axes[0, i].axis('off')
        axes[1, i].imshow(gt_rgb)
        axes[1, i].axis('off')
        axes[2, i].imshow(pred_rgb)
        axes[2, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = args.checkpoint or get_checkpoint_path()
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Expected checkpoints/best.pth (fallback: checkpoints/latest.pth)."
        )
    model = load_model(str(checkpoint_path), device=device)

    dataset = LabDataset(args.eval_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_psnr = []
    all_ssim = []
    all_lpips = []
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (L, ab) in enumerate(loader):
            L = L.to(device)
            ab = ab.to(device)
            pred_ab = model(L)
            # compute metrics on RGB images
            for i in range(L.size(0)):
                gt_lab = torch.cat([L[i], ab[i]], dim=0).unsqueeze(0)
                pred_lab = torch.cat([L[i], pred_ab[i]], dim=0).unsqueeze(0)
                gt_rgb = torch.from_numpy(lab_tensor_to_rgb(gt_lab))[0].permute(2, 0, 1).float() / 255.0
                pred_rgb = torch.from_numpy(lab_tensor_to_rgb(pred_lab))[0].permute(2, 0, 1).float() / 255.0
                all_psnr.append(psnr(gt_rgb, pred_rgb))
                all_ssim.append(ssim(gt_rgb, pred_rgb))
                all_lpips.append(lpips_metric(gt_rgb, pred_rgb))
            # Save a grid for the first batch of each epoch
            if idx == 0:
                grid_path = os.path.join(args.output_dir, f'comparison_{idx}.png')
                save_comparison_grid(L, ab, pred_ab, grid_path)

    print(f"Average PSNR: {np.mean(all_psnr):.2f}")
    print(f"Average SSIM: {np.mean(all_ssim):.4f}")
    print(f"Average LPIPS: {np.mean(all_lpips):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Lab colorization model")
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path (defaults to best, then latest)')
    parser.add_argument('--eval-data', type=str, default=str(Path(__file__).parent / 'data' / 'processed'), help='Processed validation data folder')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-dir', type=str, default='evaluation')
    args = parser.parse_args()
    evaluate(args)
