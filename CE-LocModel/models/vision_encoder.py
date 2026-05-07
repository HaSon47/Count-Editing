import torch
import torch.nn as nn
import torchvision.models as models
from models.spatial_softmax import SpatialSoftmax

class SpatialVisualEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        # Load standard ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # 1. MODIFY FIRST LAYER: Change input channels from 3 to 4 (RGB + Density)
        # We keep the original weights for RGB and initialize the Density weights
        original_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            new_conv1.weight[:, 3:, :, :] = original_conv1.weight.mean(dim=1, keepdim=True) # Init density with average
            
        resnet.conv1 = new_conv1
        
        # Remove the classification head (fc) and pooling to keep spatial features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. SPATIAL SOFTMAX: Key component from Diffusion Policy paper
        # It converts feature maps (H, W) into explicit (x, y) coordinates
        self.spatial_softmax = SpatialSoftmax(num_features=512) # ResNet18 output is 512 channels
        
        # Final projection to match conditioning size
        self.projection = nn.Linear(512 * 2, output_dim) # *2 because SpatialSoftmax gives (x,y) per channel

    def forward(self, rgb_image, density_map):
        # rgb: [B, 3, H, W], density: [B, 1, H, W]
        x = torch.cat([rgb_image, density_map], dim=1)
        features = self.backbone(x)      # Output: [B, 512, H/32, W/32]
        spatial_features = self.spatial_softmax(features) # Output: [B, 512*2]
        return self.projection(spatial_features)