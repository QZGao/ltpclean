# 0920 update: try to overfit level1-1 in one directory
# Update: Added weight saving and resume training functionality

"""
Usage Instructions:
1. Normal Training: Set resume_training = False in config/configTrain.py
   - Will load pretrained model (cfg.model_path)

2. Resume Training: Set in config/configTrain.py:
   - resume_training = True
   - resume_checkpoint_path = "ckpt/model_epoch100_20251018_19.pth"  # Specify checkpoint path to load
   - Will prioritize loading resume checkpoint, fallback to pretrained model if failed

Weight Loading Priority:
1. Resume training checkpoint (contains model weights + optimizer state + training info)
2. Pretrained model (model weights only)
3. Randomly initialized model

Saved model contains:
- Model weights (network_state_dict)
- Optimizer state (optimizer_state_dict)
- Training info (epochs, loss, model_name, etc.)

Auto Save:
- Periodic checkpoint: model_epoch{epoch}_{timestamp}.pth
"""

from models.vae.sdvae import SDVAE
from models.vae.autoencoder import AutoencoderKL
from algorithm import Algorithm
import torch
import config.configTrain as cfg
import matplotlib.pyplot as plt
import os
from datetime import datetime
from infer_test import model_test,remove_orig_mod_prefix
import logging
# Import data loading module
from dataloader.dataLoad import MarioDataset
from torch.utils.data import DataLoader

device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging():
    """Setup logging"""
    # Create logs directory
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate log filename (with timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    log_filename = f"training_log_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()  # Also output to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"init log: {log_path}")
    return logger, log_path


def save_loss_curve(loss_history, data_save_epoch, save_path="output"):
    """Save loss curve plot"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create loss curve plot
    plt.figure(figsize=(10, 6))
    x_epochs = [(i + 1) * data_save_epoch for i in range(len(loss_history))]
    plt.plot(x_epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    loss_curve_path = os.path.join(save_path, f"loss_curve_{timestamp}.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 Loss curve saved to: {loss_curve_path}")
    return loss_curve_path


# -----------------------------
# Model Saving Function
# -----------------------------
def save_model(model, optimizer, epochs, final_loss, path=cfg.ckpt_path):
    """Save trained model to ckpt directory"""

    if not os.path.exists(path):
        os.makedirs(path)

    # Generate filename (with timestamp and epoch info)
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    model_filename = f"model_epoch{epochs}_{timestamp}.pth"
    model_path = os.path.join(path, model_filename)

    # Prepare data to save
    save_data = {
        'network_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'loss': final_loss,
        'model_name': cfg.model_name,
        'batch_size': cfg.batch_size,
        'num_frames': cfg.num_frames,
        'timestamp': timestamp,
    }

    # Save model
    try:
        torch.save(save_data, model_path)
        print(f"✅ Save model to {model_path}")

    except Exception as e:
        print(f"❌ Save model failed: {e}")


def load_resume_model(model, optimizer, checkpoint_path, device_obj):
    """Load model weights and optimizer state"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')

    try:
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        ckpt = remove_orig_mod_prefix(checkpoint['network_state_dict'])
        # Load model weights
        model.load_state_dict(ckpt, strict=False)
        print("✅ Model weights loaded successfully!")

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded successfully!")
        else:
            print("⚠️ No optimizer state found in checkpoint")

        # Get training info
        start_epoch = checkpoint.get('epochs', 0)
        best_loss = checkpoint.get('loss', float('inf'))

        print(f"📊 Loaded checkpoint info:")
        print(f"   - Epoch: {start_epoch}")
        print(f"   - Loss: {best_loss:.6f}")
        print(f"   - Model: {checkpoint.get('model_name', 'Unknown')}")

        return start_epoch, best_loss

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return 0, float('inf')


def load_pretrained_model(model, device_obj):
    """Load pretrained model weights (without optimizer state)"""
    checkpoint_path = os.path.join(cfg.ckpt_path, cfg.model_path)
    if os.path.exists(checkpoint_path):
        print(f"📥 Loading diffusion forcing pretrained checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        ckpt = remove_orig_mod_prefix(state_dict["network_state_dict"])
        model.load_state_dict(ckpt, strict=False)
        print("✅ Diffusion forcing pretrained checkpoint loaded successfully!")
        return True
    else:
        print(f"⚠️ Diffusion forcing pretrained checkpoint not found: {checkpoint_path}, using random initialized model")
        return False


def vae_encode(batch_data_images, vae_model, device, scale_factor=0.7064):
    """VAE encode the images"""
    # Encode images to latent space: [batch_size, num_frames, 3, 128, 128] -> [batch_size, num_frames, 4, 32, 32]
    with torch.no_grad():
        batch_size_videos, num_frames, channels, h, w = batch_data_images.shape
        # Reshape to [batch_size * num_frames, 3, 128, 128] for batch encoding
        images_flat = batch_data_images.reshape(-1, channels, h, w).to(device)

        # VAE encoding
        if vae_model is not None:
            latent_dist = vae_model.encode(images_flat)  # [batch_size * num_frames, 3, 128, 128]
            latent_images = latent_dist.sample()  # Sample latent representation [batch_size * num_frames, 4, 32, 32]

            latent_images = latent_images * scale_factor
            # print(f"   Using scale factor: {Config.scale_factor}")

            # Reshape back to [batch_size, num_frames, 4, 32, 32]
            latent_images = latent_images.reshape(batch_size_videos, num_frames, 4, 32,
                                                  32)  # [batch_size, num_frames, 4, 32, 32]
        else:
            print("⚠️ Cannot find VAE model, use original image")
            # If no VAE, use original image directly but need to adjust shape
            latent_images = images_flat.reshape(batch_size_videos, num_frames, channels, h, w)
            print(f"   Using original image shape: {latent_images.shape}")

        # Update batch_data[0] to encoded latent representation, keep on GPU
        return latent_images


def train():
    # Initialize logging
    logger, log_path = setup_logging()

    device_obj = torch.device(device)
    
    # ==================== Performance Optimization Settings ====================
    # Enable CUDNN benchmark (optimize for fixed input sizes)
    if device == "cuda" and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("✅ CUDNN benchmark enabled")
    
    # Set matrix multiplication precision to "high" (use Tensor Cores, faster)
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
            logger.info("✅ Float32 matmul precision set to 'high'")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set matmul precision: {e}")
    # Use multi-process data loading optimization
    dataset = MarioDataset(cfg)
    dataloder = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Print dataset info (including frame skipping effect)
    logger.info(f"📊 Dataset loaded: {len(dataset)} samples")
    logger.info(f"📊 Frame sampling threshold (train_sample): {cfg.train_sample}")

    # Video sequence parameters
    num_frames = cfg.num_frames
    model_name = cfg.model_name
    loss_log_iter = cfg.loss_log_iter
    gif_save_epoch = cfg.gif_save_epoch
    checkpoint_save_epoch = cfg.checkpoint_save_epoch

    # Load complete pretrained model using Algorithm class (contains VAE and Diffusion)
    model = Algorithm(model_name, device_obj)
    model = model.to(device_obj)
    opt = model.df_model.configure_optimizers_gpt()
    scale_factor = cfg.scale_factor

    # Initialize training state
    start_epoch = 0
    best_loss = float('inf')

    # Check if need to resume training - prioritize loading resume checkpoint
    if cfg.resume_training and cfg.resume_checkpoint_path:
        print(f"🔄 Resuming training from checkpoint: {cfg.resume_checkpoint_path}")
        start_epoch, best_loss = load_resume_model(model, opt, cfg.resume_checkpoint_path, device_obj)
        if start_epoch > 0:
            print(f"✅ Resuming training from epoch {start_epoch}")
        else:
            print("⚠️ Failed to load resume checkpoint, falling back to pretrained model")
            load_pretrained_model(model, device_obj)
    else:
        # No resume training set, load pretrained model
        load_pretrained_model(model, device_obj)
        print("🆕 Starting fresh training")

    # Get VAE and Diffusion models
    vae = AutoencoderKL().to(device_obj)

    # Load your own trained VAE weights
    custom_vae_path = cfg.vae_model
    if custom_vae_path and os.path.exists(custom_vae_path):
        print(f"📥 Loading your own VAE checkpoint: {custom_vae_path}")
        custom_state_dict = torch.load(custom_vae_path, map_location=device_obj,weights_only=False)
        vae_ckpt = remove_orig_mod_prefix(custom_state_dict["network_state_dict"])
        vae.load_state_dict(vae_ckpt, strict=False)
        print("✅ Your VAE checkpoint loaded successfully!")
    else:
        print("ℹ️ Using default pre-trained VAE checkpoint")

    if vae is not None:
        vae.eval()
        for param in vae.parameters():  # Freeze VAE parameters
            param.requires_grad = False
        print("✅ VAE loaded, VAE parameters have been frozen")
    else:
        print("⚠️ Cannot find VAE model")
    
    
    # ==================== Model Compilation Optimization ====================
    # Use torch.compile to accelerate diffusion model training
    # Note: torch.compile requires PyTorch 2.0+ and CUDA
    use_compile = False
    if hasattr(torch, 'compile') and device == "cuda":
        try:
            # Compile diffusion model to accelerate training
            # mode='max-autotune' can get better performance but longer initial compilation time
            # mode='reduce-overhead' compiles faster but slightly lower performance
            compile_mode = 'max-autotune'  # Can change to 'reduce-overhead' if compilation is too slow
            print("🚀 Compiling diffusion model with torch.compile...")
            model.df_model = torch.compile(model.df_model, mode=compile_mode, backend="inductor")
            use_compile = True
            print(f"✅ Diffusion model compiled successfully with mode='{compile_mode}'!")
            logger.info(f"✅ Diffusion model compiled with mode='{compile_mode}'")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}, falling back to normal training")
            logger.warning(f"⚠️ torch.compile failed: {e}")
            use_compile = False
    elif device != "cuda":
        print("ℹ️ torch.compile requires CUDA, skipping compilation (using CPU)")
    else:
        print("ℹ️ torch.compile not available (requires PyTorch 2.0+), using normal training")
    
    epochs, batch_size = cfg.epochs, cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps

    print("---1. Start training----")
    print("---2. Load dataset---")
    total_video_sequences = len(dataset)  # Dataset has returned valid sequence count
    # Check if there is enough data
    if total_video_sequences < 1:
        print(f"❌ Dataset not enough: no valid video sequences")
        return

    num_batches = (total_video_sequences + batch_size - 1) // batch_size
    print(f"📊 Dataset info:")
    print(f"   - Valid video sequences: {total_video_sequences}")
    print(f"   - Each video has {num_frames} frames")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Batches per epoch: {num_batches}")
    print(
        f"   - Gradient accumulation steps: {gradient_accumulation_steps} (effective batch size: {batch_size * gradient_accumulation_steps})")

    # Initialize loss history
    loss_history = []
    final_avg_loss = 0  # For saving final avg_loss
    avg_loss = 0  # Initialize avg_loss to avoid UnboundLocalError

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        batch_count = 0

        # Gradient accumulation related variables
        accumulation_step = 0

        # Ensure optimizer gradients are zeroed at the start of each epoch
        opt.zero_grad()

        for batch_data in dataloder:
            batch_images, batch_actions, batch_nonterminals = batch_data
            batch_data = [
                batch_images.to(device_obj),
                batch_actions.to(device_obj),
                batch_nonterminals.to(device_obj)
            ]
            batch_data[0] = vae_encode(batch_data[0], vae, device_obj, scale_factor)

            # # Repeat batch 14 times to increase data volume
            # # over fit small dataset
            # batch_data[0] = batch_data[0].repeat(62, 1, 1, 1, 1)  # images: [batch, frames, C, H, W]
            # batch_data[1] = batch_data[1].repeat(62, 1, 1)  # actions: [batch, frames, 1]
            # batch_data[2] = batch_data[2].repeat(62, 1)  # nonterminals: [batch, frames]

            try:
                out_dict = model.df_model.training_step(batch_data)
                loss = out_dict["loss"]  # Use loss or original_loss??

                # Divide loss by accumulation steps, so gradient accumulation equals larger batch size
                loss = loss / gradient_accumulation_steps

                # Backward propagation (accumulate gradients)
                # PyTorch's backward() will add computed gradients to parameters' .grad attribute (accumulate, not replace)
                # This is why gradients automatically accumulate
                loss.backward()

                accumulation_step += 1

                # When accumulation steps reach the set value, execute optimizer update
                if accumulation_step % gradient_accumulation_steps == 0:
                    opt.step()  # Execute parameter update
                    opt.zero_grad()  # Zero gradients, prepare for next accumulation

                total_loss += loss.item() * gradient_accumulation_steps  # Correct accumulated loss value
                batch_count += 1


            except Exception as e:
                print(f"   ❌ Error in training step: {e}")
                print(f"   Batch_data shapes:")
                print(f"     images: {batch_data[0].shape}")
                print(f"     actions: {batch_data[1].shape}")
                print(f"     nonterminals: {batch_data[2].shape}")
                raise e

            # View loss and gif in batch
            if batch_count % loss_log_iter == 0:
                batch_loss = loss.item() * gradient_accumulation_steps  # Correct displayed loss value
                loss_message = f"Epoch {epoch + 1}/{epochs}, in batch: {batch_count},  Loss: {batch_loss:.6f}"
                logger.info(loss_message)

        # At end of epoch, check if there are remaining unupdated gradients
        if accumulation_step % gradient_accumulation_steps != 0:
            logger.info(
                f"Epoch {epoch + 1} ended with {accumulation_step % gradient_accumulation_steps} accumulated gradients, applying remaining update...")
            opt.step()
            opt.zero_grad()
        
        # # smalldataset
        # if batch_count > 0 and (epoch + 1) % 5 == 0:

        # large dataset
        if batch_count > 0 and (epoch + 1) % 1 == 0:
            # if batch_count > 0:
            avg_loss = total_loss / batch_count
            # scheduler.step(avg_loss)
            final_avg_loss = avg_loss  # Update final avg_loss
            # Print loss every epoch and record to history
            loss_history.append(avg_loss)  # Only record printed loss values
            loss_message = f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}"
            logger.info(loss_message)
            # Check if this is the best model, if so, save best model
            is_best = avg_loss < best_loss
            if is_best:
                # Immediately update best loss
                improvement = (best_loss - avg_loss) / best_loss if best_loss != float('inf') else 1.0
                best_loss = avg_loss
                best_message = f"This is the new best loss(improvement: {improvement:.2%})"
                logger.info(best_message)

        # Every gif_save_epoch epochs, run test once, save gif
        if (epoch + 1) % gif_save_epoch == 0:
            # # small dataset
            # model_test(cfg.test_img_path1, cfg.actions1, model, vae, device_obj, cfg.sample_step,
            #            f'{cfg.test_img_path1[-9:-4]}_epoch{epoch + 1}_r', epoch=epoch + 1, output_dir=cfg.out_dir)
            # model_test(cfg.test_img_path2, cfg.actions3, model, vae, device_obj, cfg.sample_step,
            #            f'{cfg.test_img_path2[-9:-4]}_epoch{epoch + 1}_rj', epoch=epoch + 1, output_dir=cfg.out_dir)

            # large data
            model_test(cfg.test_img_path1, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path1[-9:-4]}_epoch{epoch + 1}_r', epoch=epoch + 1, output_dir=cfg.out_dir)
            model_test(cfg.test_img_path1, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path1[-9:-4]}_epoch{epoch + 1}_rj', epoch=epoch + 1, output_dir=cfg.out_dir)
            model_test(cfg.test_img_path2, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path2[-9:-4]}_epoch{epoch + 1}_r', epoch=epoch + 1, output_dir=cfg.out_dir)
            model_test(cfg.test_img_path2, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path2[-9:-4]}_epoch{epoch + 1}_rj', epoch=epoch + 1, output_dir=cfg.out_dir)
            model_test(cfg.test_img_path3, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path3[-9:-4]}_epoch{epoch + 1}_r', epoch=epoch + 1, output_dir=cfg.out_dir)
            model_test(cfg.test_img_path3, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path3[-9:-4]}_epoch{epoch + 1}_rj', epoch=epoch + 1, output_dir=cfg.out_dir)
            model_test(cfg.test_img_path4, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path4[-9:-4]}_epoch{epoch + 1}_r', epoch=epoch + 1, output_dir=cfg.out_dir)
            model_test(cfg.test_img_path4, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                       f'{cfg.test_img_path4[-9:-4]}_epoch{epoch + 1}_rj', epoch=epoch + 1, output_dir=cfg.out_dir)

        # Every checkpoint_save_epoch epochs, save checkpoint once
        if (epoch + 1) % checkpoint_save_epoch == 0:
            current_loss = avg_loss if batch_count > 0 else 0
            save_model(model, opt, epoch + 1, current_loss, path=cfg.ckpt_path)
            checkpoint_message = f"💾 Checkpoint saved at epoch {epoch + 1}"
            logger.info(checkpoint_message)

    completion_message = "Training completed!"
    print(completion_message)
    logger.info(completion_message)

    # Save final model after training completes
    if epochs >= 1 and final_avg_loss > 0:
        save_message = "💾 Saving final training model..."
        print(save_message)
        logger.info(save_message)

        save_model(model, opt, epochs, final_avg_loss, path=cfg.ckpt_path)

        # Record training statistics
        stats_message = f"📊 Training statistics: total epochs: {epochs}, best loss: {best_loss:.6f}, final loss: {final_avg_loss:.6f}, total batches: {batch_count * epochs}"
        print(f"📊 Training statistics:")
        print(f"    Total epochs: {epochs}")
        print(f"    Best loss: {best_loss:.6f}")
        print(f"    Final loss: {final_avg_loss:.6f}")
        print(f"    Total batches: {batch_count * epochs}")
        logger.info(stats_message)
        # # small dataset
        # model_test(cfg.test_img_path1, cfg.actions1, model, vae, device_obj, cfg.sample_step,
        #            f'{cfg.test_img_path1[-9:-4]}_epoch{epoch + 1}_r', epoch='result', output_dir=cfg.out_dir)
        # model_test(cfg.test_img_path3, cfg.actions3, model, vae, device_obj, cfg.sample_step,
        #            f'{cfg.test_img_path3[-9:-4]}_epoch{epoch + 1}_rj', epoch='result', output_dir=cfg.out_dir)

        # large dataset
        model_test(cfg.test_img_path1, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path1[-9:-4]}_result_{epochs}_r', epoch='result', output_dir=cfg.out_dir)
        model_test(cfg.test_img_path1, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path1[-9:-4]}_result_{epochs}_rj', epoch='result', output_dir=cfg.out_dir)
        model_test(cfg.test_img_path2, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path2[-9:-4]}_result_{epochs}_r', epoch='result', output_dir=cfg.out_dir)
        model_test(cfg.test_img_path2, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path2[-9:-4]}_result_{epochs}_rj', epoch='result', output_dir=cfg.out_dir)
        model_test(cfg.test_img_path3, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path3[-9:-4]}_epoch{epoch + 1}_r', epoch='result', output_dir=cfg.out_dir)
        model_test(cfg.test_img_path3, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path3[-9:-4]}_epoch{epoch + 1}_rj', epoch='result', output_dir=cfg.out_dir)
        model_test(cfg.test_img_path4, cfg.actions1, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path4[-9:-4]}_epoch{epoch + 1}_r', epoch=epoch + 1, output_dir=cfg.out_dir)
        model_test(cfg.test_img_path4, cfg.actions2, model, vae, device_obj, cfg.sample_step,
                   f'{cfg.test_img_path4[-9:-4]}_epoch{epoch + 1}_rj', epoch=epoch + 1, output_dir=cfg.out_dir)

    # Save final loss curve to output directory
    if len(loss_history) > 0:
        final_loss_curve_path = save_loss_curve(loss_history, 1, save_path="output")
        logger.info(f"Final loss curve saved to: {final_loss_curve_path}")

    # Record log file path
    final_log_message = f"Log path: {log_path}"
    print(final_log_message)
    logger.info(final_log_message)


if __name__ == "__main__":
    train()