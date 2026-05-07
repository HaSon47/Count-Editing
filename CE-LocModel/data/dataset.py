import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class ObjectPlacementDataset(Dataset):
    def __init__(self, root_dir, target_size=(512, 512)):
        """
        Args:
            root_dir (str): Path to the specific split folder (e.g. 'data/train').
                            Must contain 'images', 'density', and 'annotation' subfolders.
            target_size (tuple): Model input size (width, height).
        """
        self.root_dir = root_dir
        self.target_size = target_size
        
        self.image_dir = os.path.join(root_dir, 'images')
        self.density_dir = os.path.join(root_dir, 'density')
        self.annot_dir = os.path.join(root_dir, 'annotation')
        
        # Get all valid image filenames
        self.files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.files.sort() # Ensure consistent order

    def resize_and_pad(self, img, density):
        w, h = img.size
        target_w, target_h = self.target_size
        
        # 1. Calculate Scale
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 2. Resize
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        density = density.resize((new_w, new_h), resample=Image.NEAREST)
        
        # 3. Pad (Top-Left alignment)
        padded_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        padded_img.paste(img, (0, 0))
        
        padded_density = Image.new("L", (target_w, target_h), 0)
        padded_density.paste(density, (0, 0))
        
        return padded_img, padded_density, scale

    def parse_annotation(self, filename):
        # Change extension to .json
        json_name = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(self.annot_dir, json_name)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        class_name = data['class']
        # bbox is already [center_x, center_y, w, h]
        if 'target_bbox' not in data:
            bbox = [0.0, 0.0, 0, 0] # Default bbox if not provided
        else:
            bbox = data['target_bbox'] 
        
        return class_name, bbox

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        
        # 1. Load Files
        img_path = os.path.join(self.image_dir, filename)
        # Density is typically .png
        density_name = os.path.splitext(filename)[0] + '.png'
        density_path = os.path.join(self.density_dir, density_name)
        
        raw_img = Image.open(img_path).convert("RGB")
        raw_density = Image.open(density_path).convert("L")
        
        # 2. Get Annotation
        class_name, bbox = self.parse_annotation(filename)
        
        # 3. Process Images
        img, density, scale = self.resize_and_pad(raw_img, raw_density)
        
        # 4. Process BBox
        # Input Format: [center_x, center_y, w, h] (Absolute Pixels)
        cx, cy, w, h = bbox
        
        # Apply scaling (Coordinate Transformation)
        # Since we pad at (0,0), we simply scale the center coordinates
        cx *= scale
        cy *= scale
        w *= scale
        h *= scale
        
        # Normalize to [-1, 1] relative to TARGET SIZE
        target_w, target_h = self.target_size
        
        norm_cx = (cx / target_w) * 2 - 1
        norm_cy = (cy / target_h) * 2 - 1
        norm_w = (w / target_w) * 2 - 1
        norm_h = (h / target_h) * 2 - 1
        
        norm_box = torch.tensor([norm_cx, norm_cy, norm_w, norm_h], dtype=torch.float32)

        return {
            "pixel_values": F.to_tensor(img),       # [3, H, W]
            "density_map": F.to_tensor(density),    # [1, H, W]
            "text": class_name,                     # Str
            "bbox": norm_box,                       # [4]
            "scale": scale,                         # Float
            "original_size": raw_img.size           # (W, H)
        }