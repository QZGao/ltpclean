from torchvision.transforms import InterpolationMode
from infer_test import remove_orig_mod_prefix
from models.vae.sdvae import SDVAE
from models.vae.autoencoder import AutoencoderKL
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import config.configTrain as cfg

def calculate_psnr(img1, img2):
    """计算两张图像之间的PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
def get_img_data(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.Resize((128, 128),interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def get_web_img(img):
    # img.shape = [c, h, w] 3,256,256
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    img_3ch = np.clip(img_3ch*0.5+0.5, 0, 1)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    return img_3ch
def decode():
  out_dir='decode_test/'
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)

  data_path = 'datatrain/'
  image_paths = []
  for root, dirs, files in os.walk(data_path):
      for file in files:  # 遍历files列表，不是root
          if file.lower().endswith('.png'):
              file_path = os.path.join(root, file)
              image_paths.append(file_path)

  device ="cuda:0"
  vae = AutoencoderKL().to(device)
  custom_vae_path = cfg.vae_model
  if custom_vae_path and os.path.exists(custom_vae_path):
      print(f"📥 load your own vae ckpt: {custom_vae_path}")
      custom_state_dict = torch.load(custom_vae_path, map_location=device,weights_only=False)
      ckpt = remove_orig_mod_prefix(custom_state_dict['network_state_dict'])
      vae.load_state_dict(ckpt, strict=False)
      print("✅ your vae ckpt loaded successfully！")
  else:
      print("ℹ️ use default pre-trained vae ckpt")
  vae.eval()
  psnr_list = []
  with torch.no_grad():
      for p in image_paths:
          img = get_img_data(p).to(device)
          latent = vae.encode(img)
          latent = latent.sample()
          decode = vae.decode(latent)
          decoded_img_array = get_web_img(decode[0].cpu().numpy())
          
          # 计算PSNR
          orig_img = Image.open(p).convert('RGB').resize((cfg.image_size, cfg.image_size))
          orig_array = np.array(orig_img)
          psnr = calculate_psnr(orig_array, decoded_img_array)
          psnr_list.append(psnr)
          print(f"{os.path.basename(p)}: PSNR = {psnr:.2f} dB")
          
          decoded_img = Image.fromarray(decoded_img_array)
          decoded_img.save(f"{out_dir}{os.path.basename(p)}")
  
  print(f"\n平均PSNR: {np.mean(psnr_list):.2f} dB")

if __name__=='__main__':
  decode()
