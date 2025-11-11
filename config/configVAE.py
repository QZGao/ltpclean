"""Local Web"""
"""sd 1.5 512*512 VAE"""

out_dir: str = "./output/VAE"
data_path: str = "/content/drive/MyDrive/mario_data"
ckpt_path: str = "./ckpt/VAE"
model_path: str = ""

"""Resume Training Config"""
resume_training = False  # 是否继续训练
resume_checkpoint_path = "/content/drive/MyDrive/my_models/1026largeDATA_df/model_epoch110_20251028_23.pth"  # 继续训练的checkpoint路径，例如: "ckpt/model_epoch100_20251018_19.pth"


"""Train Config"""
lr = 1e-4
img_size = 128
img_channel = 3
latent_ch: int = 4
latent_size: int = 32
num_workers_folders=12
num_workers = 12

loss_log_iter: int = 20  # loss数据print和保存至log日志的间隔 \log

img_save_epoch: int = 6  # avgloss和gif保存间隔 \output

checkpoint_save_epoch: int = 6  # checkpoint保存间隔


batch_size: int = 64
epochs: int = 30          # 测试epoch数量


test_img_path: str = "./eval_data/vae"