# 数据加载模块 - 优化的大数据集处理
# 包含MarioDataset类和相关的视频序列构建函数

from typing import Optional
import re
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

from torchvision.transforms import InterpolationMode


class MarioDataset(Dataset):
    """load mario dataset __init__ action and img paths,
     __getitem__  will return image"""
    """up to date: 1110 return image for vae train"""
    def __init__(self, cfg):
        self.data_path = cfg.data_path
        self.image_size = cfg.img_size
        self.num_workers_folders = cfg.num_workers_folders
        self.image_files = [] # image files path (xxx.png)
        self._load_data()
        image_size = cfg.img_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ])

    def _load_data(self):
        """load all png files and corresponding actions - optimized for large datasets"""
        print(f" data path is scanning: {self.data_path}")
        if not os.path.exists(self.data_path): 
            print(f"❌ data path not found: {self.data_path}")
            return
        
        # 使用多进程扫描文件
        import multiprocessing as mp
        
        # 收集所有子目录
        subdirs = []
        for root, dirs, files in os.walk(self.data_path):
            if root != self.data_path and files:  # 跳过根目录，只处理有文件的子目录
                subdirs.append(root)
        
        print(f"Found {len(subdirs)} subdirectories to scan")
        
        # 并行处理每个子目录，每个子目录内已按帧号排序
        with ProcessPoolExecutor(max_workers=self.num_workers_folders) as executor:
            futures = [executor.submit(MarioDataset._scan_directory, subdir) for subdir in subdirs]
            
            for future in futures:
                files= future.result()
                self.image_files.extend(files)
        print(f"✅ Loaded {len(self.image_files)} valid images from {len(subdirs)} levels")
    
    @staticmethod
    def _scan_directory(directory):
        """扫描单个目录，返回文件路径和动作，按帧号排序"""
        all_files = []
        for file in os.listdir(directory):
            if file.lower().endswith('.png'):
                file_path = os.path.join(directory, file)

                all_files.append(file_path)
        return all_files

    def __len__(self):
        """返回图片数量"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """get the data sample of the specified index - optimized for large datasets
        """
        # 构建单个图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image



