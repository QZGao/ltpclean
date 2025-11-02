import importlib
import os.path as osp
import numpy as np
import torch
from config.Config_VAE import (Config)
from config.configTrain import *
from algorithm import Algorithm
import random
from einops import rearrange
import time
import cv2

def read_model(model_name, model_path, device='cpu'):
    model = Algorithm(model_name,device)
    state_dict = torch.load(
        osp.join("ckpt", model_path),
        map_location=torch.device(device),weights_only=False
    )
    model.load_state_dict(state_dict['network_state_dict'],strict=False)
    model.eval().to(device)
    return model

def read_file(data_file, data_type="java"):
    print("read_file", data_type)
    data_with_nan = np.genfromtxt(data_file, delimiter=',')
    
    # 处理一维数组（单行数据）
    if data_with_nan.ndim == 1:
        data_with_nan = data_with_nan.reshape(1, -1)
    
    np_data = data_with_nan[:, ~np.isnan(data_with_nan).any(axis=0)]
    
    # 检查数据格式：256x256格式的数据有 256*256*3+1 = 196609 列
    # 如果是新格式（256x256），则不应用切片，直接返回所有列
    expected_cols_256 = img_size * img_size * 3 + 1  # 196609 for 256x256
    expected_cols_128 = Config.resolution * Config.resolution * 3 + 1  # 49153 for 128x128
    
    if np_data.shape[1] == expected_cols_256:
        # 新格式（256x256），不应用切片
        print(f"Detected 256x256 format ({expected_cols_256} columns), skipping slice")
        data = np_data
    elif np_data.shape[1] >= expected_cols_128:
        # 旧格式（128x128或更复杂格式），应用切片
        print(f"Detected old format, applying slice: {Config.data_start_end}")
        data = np_data[:, Config.data_start_end[0]:Config.data_start_end[1]]
    else:
        # 未知格式，直接返回（可能是新格式但没有预期列数，或者格式异常）
        print(f"Warning: Unknown data format with {np_data.shape[1]} columns, returning as-is")
        data = np_data
    
    return data

def split_data_mario(epi_data, horizon_len, device):
    """
    处理mario数据，与 infer_test.py 中的 get_img_data 处理逻辑保持一致：
    - 原始像素值 [0, 255] -> ToTensor [0, 1] -> Normalize [-1, 1]
    """
    episode_len, _ = epi_data.shape
    assert episode_len % horizon_len == 0
    data_tensor = torch.tensor(epi_data, dtype=torch.float32, device=device)
    # 重塑为 [episode_len, H, W, C]
    cur_img = data_tensor[:, :-1].reshape(episode_len, img_size, img_size, 3)
    # 转换为 [episode_len, C, H, W]
    cur_img = cur_img.permute(0, 3, 1, 2)
    cur_img = cur_img.reshape(episode_len, 3, img_size, img_size)
    # 归一化到 [0, 1] (ToTensor的等价操作)
    cur_img = cur_img / 255.0
    # 归一化到 [-1, 1] (与get_img_data中的Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))一致)
    # Normalize公式: (x - mean) / std = (x - 0.5) / 0.5 = 2*x - 1
    cur_img = cur_img * 2.0 - 1.0
    cur_action = data_tensor[:, -1].reshape(episode_len, 1).long()
    dict_data = {
        "epi_len": episode_len,
        "cur_img": cur_img,
        "cur_act_int": cur_action,
    }
    return dict_data

def process_npdata(np_data, horizon_len, start_idx, device):
    file_data_len = np_data.shape[0]
    end_idx = start_idx + horizon_len
    if end_idx > file_data_len:
        end_idx = file_data_len
        start_idx = end_idx - horizon_len
    epi_data = np_data[start_idx:end_idx]
    dict_data = split_data_mario(epi_data, horizon_len, device)
    batch = {}
    for k, v in dict_data.items():
        if torch.is_tensor(v):
            batch[k] = v
    batch_data = data_formater(batch, horizon_len)
    batch_data["start_idx"] = start_idx
    return batch_data

def data_formater(data_dict, epi_len):
    batch = {}
    ori_obs = data_dict["cur_img"]
    batch["observations"] = rearrange(ori_obs, '(b t) c h w -> b t c h w', t=epi_len)
    batch["cur_actions"] = rearrange(data_dict["cur_act_int"], '(b t) e -> b t e', t=epi_len, e=1)
    return batch

def init_simulator(model, batch):
    obs = batch["observations"]
    obs = rearrange(obs, 'b t c h w -> (b t) c h w')
    with torch.no_grad():
        init_zeta = model.init_wm(obs[0:1])
        return obs, init_zeta

def get_web_img(img):
    """
    将图像从模型输出格式转换为web显示格式
    输入: img.shape = [c, h, w]，值范围 [-1, 1] (与get_img_data的Normalize一致)
    输出: [h, w, c]，值范围 [0, 255]，uint8
    """
    # img.shape = [c, h, w]
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    # 从 [-1, 1] 反归一化到 [0, 1]: (x + 1) / 2
    img_3ch = (img_3ch + 1.0) / 2.0
    img_3ch = np.clip(img_3ch, 0, 1)
    img_3ch = cv2.resize(img_3ch, (300, 300), interpolation=cv2.INTER_LINEAR)
    # img_3ch = cv2.resize(img_3ch, (25, 25), interpolation=cv2.INTER_LINEAR)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    return img_3ch
    
global np_data
np_data = read_file(file_path, data_type)
def get_data(if_random=False):
    start_time = time.time()
    start_idx = 0
    if if_random:
        start_idx = random.randint(0, np_data.shape[0])
    batch_data = process_npdata(np_data, 1, start_idx, device)
    return batch_data

if __name__ == "__main__":
    # model = read_model(model_name, model_path, device)
    device = 'cpu'

    np_data = read_file(file_path, data_type)

    start_time = time.time()
    start_idx = random.randint(0, np_data.shape[0])
    start_idx = 0
    batch_data = process_npdata(np_data, 1, start_idx, device)
    end_time = time.time()
    print(f"process_npdata cost time: {end_time - start_time:.4f} s")
    start_time = time.time()
    data = batch_data["observations"]

    # obs, wm = init_simulator(model, batch_data)
    img = get_web_img(data[0, 0].cpu().numpy())  # data shape: [b, t, c, h, w], need [c, h, w]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./eval_data/init.jpg', img)
    print(img.shape)

