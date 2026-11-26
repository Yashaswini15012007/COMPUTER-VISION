from pathlib import Path

import torch

processed_dir = Path(__file__).parent / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

num_samples = 200
size = 256
for i in range(num_samples):
    # Random L channel in [0,1]
    L = torch.rand(1, size, size)
    # Random ab channels in [-1,1]
    ab = torch.rand(2, size, size) * 2 - 1
    tensor = torch.cat([L, ab], dim=0)
    torch.save(tensor, processed_dir / f'dummy_{i:04d}.pt')
print(f'Generated {num_samples} dummy tensors in {processed_dir}')
