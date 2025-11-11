# 0920 update: try to overfit level1-1 in one directory


import torch

import config.configTrain as cfg


# 导入数据加载模块
from dataloader.dataLoad import MarioDataset

device: str = "cuda" if torch.cuda.is_available() else "cpu"



def train():

    device_obj = torch.device(device)
    # 使用多进程数据加载优化
    dataset = MarioDataset(cfg)

    # video sequence parameters
    num_frames = 13
    frame_interval = 13

    epochs, batch_size = 1, 2


    print("---2. load dataset---")
    total_samples = len(dataset)
    # 检查是否有足够的数据
    if total_samples < num_frames:
        print(f"❌ dataset not enough: need at least {num_frames} samples, but only {total_samples} samples")
        return
    # 计算可以创建多少个完整的视频序列
    num_videos = (total_samples - num_frames) // frame_interval + 1
    print(f"dataset loaded: {total_samples} samples, construct {num_videos} complete video sequences, "
          f"each video has {num_frames} frames, construct {(num_videos + batch_size - 1) // batch_size} batches, the batch size is {batch_size}")



    # 预计算所有有效的视频序列起始位置,间隔一个frame_interval取一个video sequence, 最终剩下不足一个video的扔掉
    valid_starts = []
    for start in range(0, total_samples - num_frames + 1, frame_interval):
        valid_starts.append(start)

    # 按batch_size分组处理
    num_valid_videos = len(valid_starts)



if __name__ == "__main__":
    train()