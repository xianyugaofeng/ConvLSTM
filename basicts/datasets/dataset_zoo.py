import os
import numpy as np
import pandas as pd
import gzip, random
from torch.utils.data import Dataset

class MovingMNISTDataset(Dataset):
    def __init__(self, config, split='learn'):
        super().__init__()
        with gzip.open(config.get('DATA_FILE_PATH'), 'rb') as f:            
            """
            将字节流解释为一维的 uint8 数组 跳过文件开头的 16 个字节
            因为原始数据文件类似 MNIST 格式有一个16字节的头部，包含魔数、样本数量、行数、列数等信息
            MovingMNIST的每一帧是64×64灰度图，像素值0~255。每个像素在文件中占1字节
            MovingMNIST生成时需要从MNIST取数字图像
            """
            # f.read()：读取整个文件的字节流 跳过文件开头的 16 个字节
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            self.datas = self.datas.reshape(-1, *config.get("IMAGE_SIZE"))
            # 重塑为(样本数, 时间步, 高度, 宽度)的形式