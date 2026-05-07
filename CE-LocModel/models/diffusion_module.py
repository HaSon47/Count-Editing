import torch
import torch.nn as nn
from models.vision_encoder import SpatialVisualEncoder
from models.text_encoder import CLIPTextEncoder 
from models.noise_pred_net import ConditionalUnet1D

class ObjectPlacementPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # --- CONFIG LOADING ---
        vis_dim = cfg['vision_encoder']['output_dim']
        text_dim = cfg['text_encoder']['output_dim']
        
        unet_dims = cfg['noise_net']['down_dims']
        unet_kernel = cfg['noise_net']['kernel_size']
        cond_dim = vis_dim + text_dim
        
        # 1. Encoders
        self.vision_encoder = SpatialVisualEncoder(output_dim=vis_dim)
        
        self.text_encoder = CLIPTextEncoder(
            model_name=cfg['text_encoder']['model_name'],
            output_dim=text_dim,
            freeze_backbone=cfg['text_encoder']['freeze']
        )
        
        # 2. U-Net
        self.noise_net = ConditionalUnet1D(
            input_dim=cfg['noise_net']['input_dim'], 
            global_cond_dim=cond_dim,
            down_dims=unet_dims,
            kernel_size=unet_kernel,
            n_groups=cfg['noise_net']['n_groups']
        )
        
        # 3. NOISE SCHEDULER SETUP (DDPM)
        # This was missing in the previous version!
        if 'diffusion' in cfg:
            self.num_timesteps = cfg['diffusion']['num_timesteps']
        else:
            self.num_timesteps = 100
        self.setup_noise_schedule()
        print(f"Diffusion noise schedule set up with {self.num_timesteps} timesteps.")

    def setup_noise_schedule(self):
        """
        Defines the linear beta schedule and pre-calculates alpha_bar.
        """
        # Standard parameters for DDPM
        beta_start = 0.0001
        beta_end = 0.02
        
        # 1. Betas (Linear Schedule)
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        
        # 2. Alphas = 1 - Betas
        alphas = 1.0 - betas
        
        # 3. Alpha Cumulative Product (Alpha Bar)
        # This represents how much "signal" remains at step t
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as a buffer so it moves to GPU automatically with the model
        # but is NOT treated as a trainable parameter.
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def get_alpha_bar(self, t):
        """
        Retrieves alpha_bar for a batch of timesteps t.
        Returns shape [Batch, 1] for broadcasting.
        """
        # Gather values at indices t
        # alpha_cumprod is shape [100], t is shape [B]
        acc_alpha = self.alphas_cumprod[t] 
        
        # Reshape to [B, 1] so we can multiply with BBox [B, 4]
        return acc_alpha.unsqueeze(-1)

    def compute_loss(self, rgb, density, text, gt_bbox):
        # gt_bbox shape: [Batch, 4]
        
        # 1. Encode Conditions
        vis_emb = self.vision_encoder(rgb, density) # [B, 128]
        text_emb = self.text_encoder(text)          # [B, 128]
        global_cond = torch.cat([vis_emb, text_emb], dim=-1) # [B, 256]
        
        # 2. Sample Noise and Timestep
        B = rgb.shape[0]
        # Random timestep for each item in batch
        t = torch.randint(0, self.num_timesteps, (B, ), device=rgb.device)
        noise = torch.randn_like(gt_bbox)
        
        # 3. Add Noise (Forward Diffusion)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        alpha_bar_t = self.get_alpha_bar(t)
        noisy_bbox = torch.sqrt(alpha_bar_t) * gt_bbox + torch.sqrt(1 - alpha_bar_t) * noise
        
        # 4. Predict Noise
        # Reshape for U-Net: [Batch, 4] -> [Batch, 1, 4] (Horizon=1)
        noisy_bbox_input = noisy_bbox.unsqueeze(1) 
        
        noise_pred = self.noise_net(
            sample=noisy_bbox_input, 
            timestep=t, 
            global_cond=global_cond
        )
        
        # Remove horizon dim: [Batch, 1, 4] -> [Batch, 4]
        noise_pred = noise_pred.squeeze(1) 
        
        # 5. Loss (MSE between predicted noise and actual noise)
        return torch.nn.functional.mse_loss(noise_pred, noise)