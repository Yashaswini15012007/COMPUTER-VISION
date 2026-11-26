import numpy as np
import torch
import torch.nn.functional as F
from skimage import color, io
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Shared checkpoint configuration (used across training, inference, evaluation,
# and the Streamlit UI). All components will reference these constants so the
# file locations stay in sync.
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_LATEST = CHECKPOINT_DIR / "latest.pth"
CHECKPOINT_BEST = CHECKPOINT_DIR / "best.pth"


def get_checkpoint_path() -> Path | None:
    if CHECKPOINT_BEST.exists():
        return CHECKPOINT_BEST
    if CHECKPOINT_LATEST.exists():
        return CHECKPOINT_LATEST
    return None

def rgb_to_lab_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert a NumPy RGB image (H, W, 3) to a LAB tensor (3, H, W).
    The L channel is normalized to [0, 1], ab channels to [-1, 1].
    """
    lab = color.rgb2lab(img)
    L = lab[..., 0] / 100.0
    ab = lab[..., 1:] / 128.0
    L = torch.from_numpy(L).unsqueeze(0).float()
    ab = torch.from_numpy(ab).permute(2, 0, 1).float()
    return torch.cat([L, ab], dim=0)

def lab_tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a LAB tensor (3, H, W) or batch (B, 3, H, W) to RGB.
    Assumes the same normalization used in `rgb_to_lab_tensor`.
    """
    batched = tensor.ndim == 4
    if not batched:
        tensor = tensor.unsqueeze(0)
    lab = tensor.detach().cpu().permute(0, 2, 3, 1).numpy().copy()
    lab[..., 0] *= 100.0
    lab[..., 1:] *= 128.0
    rgb = np.clip(color.lab2rgb(lab), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb if batched else rgb[0]

def normalize_l_channel(L: torch.Tensor) -> torch.Tensor:
    """Normalize a single‑channel L tensor to [0, 1]."""
    return L / 100.0

def denormalize_l_channel(L_norm: torch.Tensor) -> torch.Tensor:
    """Reverse the L‑channel normalization."""
    return L_norm * 100.0

def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak Signal‑to‑Noise Ratio between two tensors (C, H, W)."""
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse)).item()

def ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Structural Similarity Index (a simple implementation for tensors)."""
    from skimage.metrics import structural_similarity as ssim_fn
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    return ssim_fn(img1_np, img2_np, data_range=data_range, multichannel=True)

def lpips_metric(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Placeholder for LPIPS metric – actual implementation requires the LPIPS package.
    Returns 0.0 if the package is not installed.
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='vgg')
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        img1 = (img1 * 2) - 1
        img2 = (img2 * 2) - 1
        return loss_fn(img1, img2).item()
    except Exception:
        return 0.0
