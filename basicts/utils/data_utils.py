import numpy as np
import torch

def set_random_seed(seed):
    np.random.seed(seed) # NumPy的随机数生成器
    torch.manual_seed(seed) # 固定PyTorch在CPU上的随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # 固定所有GPU上的随机数生成器