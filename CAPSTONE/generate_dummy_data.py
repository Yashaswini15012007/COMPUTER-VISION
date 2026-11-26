import torch
from pathlib import Path
import numpy as np
import os

PROCESSED_DIR = Path(__file__).parent / 'colorizer' / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

num_samples = 100
size = 256
for i in range(num_samples):
    # Random L channel in [0,1]
    L = torch.rand(1, size, size)
    # Random ab channels in [-1,1]
    ab = torch.rand(2, size, size) * 2 - 1
    tensor = torch.cat([L, ab], dim=0)
    torch.save(tensor, PROCESSED_DIR / f'dummy_{i:04d}.pt')
print(f'Generated {num_samples} dummy tensors in {PROCESSED_DIR}')
