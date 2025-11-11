"""
将图像转换为 0-frameArray.txt 格式
根据 infer_test.py 中的 get_img_data 函数处理逻辑：
- Resize 到 256x256 (NEAREST插值)
- 转换为RGB格式
- 保存为原始像素值 0-255

txt 文件格式：
- 每行：图像像素值（resolution*resolution*3个，按HWC顺序展平） + 动作值（1个）
- 像素值范围：0-255
- 分辨率：256x256
"""
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
import config.configTrain as cfg


def convert_image_to_txt(image_path, output_path, action_value=0, resolution=None):
    """
    将图像转换为 txt 文件格式
    
    Args:
        image_path: 输入图像路径 (如 'eval_data/demo1.png')
        output_path: 输出 txt 文件路径 (如 'eval_data/0-frameArray.txt')
        action_value: 动作值，默认为 0（无动作）
        resolution: 目标分辨率，默认使用 cfg.img_size (256)
    """
    # 使用指定的分辨率，或从配置文件中读取
    if resolution is None:
        resolution = cfg.img_size  # 默认256
    
    # 1. 读取图像并转换为 RGB
    print(f"[读取图像] {image_path}")
    img = Image.open(image_path).convert('RGB')
    
    # 2. 调整大小为指定分辨率（使用NEAREST插值，与get_img_data一致）
    print(f"[调整图像大小] {img.size} -> ({resolution}, {resolution}) (使用NEAREST插值)")
    img = img.resize((resolution, resolution), Image.NEAREST)
    
    # 3. 转换为 numpy 数组，形状为 (H, W, C) = (resolution, resolution, 3)
    img_array = np.array(img, dtype=np.uint8)
    print(f"[图像数组形状] {img_array.shape} (应该是 ({resolution}, {resolution}, 3))")
    
    # 4. 展平为一行，按 HWC 顺序：[h, w, c] -> [resolution*resolution*3]
    # 这与 split_data_mario 中的 reshape(episode_len, resolution, resolution, 3) 对应
    img_flat = img_array.flatten()  # 默认按 C-order，即行优先展平
    # np.array 从 PIL 得到的已经是 HWC 格式，flatten() 会按行优先（C-order）展平
    # 这正好符合 split_data_mario 中的 reshape 逻辑
    
    expected_size = resolution * resolution * 3
    print(f"[展平后形状] {img_flat.shape} (应该是 ({expected_size},))")
    
    # 5. 添加动作值
    row_data = np.append(img_flat, action_value)
    print(f"[最终数据形状] {row_data.shape} (应该是 ({expected_size + 1},))")
    
    # 6. 保存为 CSV 格式（逗号分隔）
    print(f"[保存到] {output_path}")
    np.savetxt(output_path, row_data.reshape(1, -1), delimiter=',', fmt='%d')
    
    print(f"[转换完成]")
    print(f"   输入: {image_path}")
    print(f"   输出: {output_path}")
    print(f"   图像分辨率: {resolution}x{resolution}")
    print(f"   像素值范围: {img_flat.min()}-{img_flat.max()}")
    print(f"   动作值: {action_value}")


if __name__ == "__main__":
    # 转换 demo1.png 为 0-frameArray.txt
    # 使用 256x256 分辨率（与 VAE 输入一致）
    convert_image_to_txt(
        image_path='eval_data/demo1.png',
        output_path='eval_data/0-frameArray1.txt',
        action_value=0,  # 可以根据需要修改动作值
        resolution=126   # VAE 输入分辨率
    )

