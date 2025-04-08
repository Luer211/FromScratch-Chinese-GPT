import os
import torch
import numpy as np

def load_bin_data(split, data_dir="data"):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.fromfile(path, dtype=np.uint16)
    return torch.tensor(data, dtype=torch.long)

def get_batch(data_tensor, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
