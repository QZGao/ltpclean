# compute scaling factor of trained VAE to train diffusion model

from infer_test import remove_orig_mod_prefix
from models.vae.autoencoder import AutoencoderKL
import torch
import config.configVAE as cfg
from torch.utils.data import DataLoader
import os
from dataloader.dataLoadvae import MarioDataset

device: str = "cuda" if torch.cuda.is_available() else "cpu"

def estimate_scaling_factor():
    target_std = 1.0

    device_obj = torch.device(device)
    batch_size = cfg.batch_size
    dataset = MarioDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)
    model = AutoencoderKL().to(device_obj)
    
    if cfg.model_path:
        ckpt_path = os.path.join(cfg.ckpt_path, cfg.model_path)
    else:
        ckpt_path = None
    
    if os.path.exists(ckpt_path):
        print(f"📥 Loading pretrained checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
        ckpt = remove_orig_mod_prefix(state_dict['network_state_dict'])
        model.load_state_dict(ckpt, strict=False)
        print("✅ Checkpoint loaded successfully")
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_path}, using initialized model")
    
    model.eval()
    all_latents = []
    total_samples = len(dataset)


    print(f"📊 Computing scaling factor using {total_samples} samples")
    print(f"📊 Batch size: {batch_size}")
    count = 0

    with torch.no_grad():
        for batch_img in dataloader:
            batch_img = batch_img.to(device_obj)
            encoded = model.encode(batch_img).sample()
            all_latents.append(encoded.cpu())
            count += 1
            
            if count % 50 == 0:
                print(f"Processed {count} batches")
    
    all_latents = torch.cat(all_latents, dim=0)
    global_std = all_latents.std().item()
    scaling = target_std / global_std
    
    print(f"📊 Computed scaling factor:")
    print(f"   Global latent std: {global_std:.6f}")
    print(f"   Target std: {target_std:.6f}")
    print(f"   Recommended scaling factor: {scaling:.6f}")
    
    return scaling




if __name__ == "__main__":
    estimate_scaling_factor()