import os
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from models.diffusion_module import ObjectPlacementPolicy
from data.dataset import ObjectPlacementDataset
from utils.visualization import visualize_result 
from train import load_config

def save_pred_box(pred_boxes, scale, save_path, original_size=(512, 512)):
    # This function is now replaced by visualize_result which also saves the image with the box drawn.
    # If you want to save just the box coordinates, you can implement that here.
    # 2. Denormalize Coordinates
    # unpack normalized box (Range: -1 to 1)
    processed_boxes = []
    for pred_box in pred_boxes:
        x_norm, y_norm, w_norm, h_norm = pred_box
        target_w, target_h = original_size
        
        # Shift to [0, 1] then scale to the Model Input Dimension
        # (We use target_w for x/w and target_h for y/h)
        x_in_model = ((x_norm + 1) / 2) * target_w
        y_in_model = ((y_norm + 1) / 2) * target_h
        w_in_model = ((w_norm + 1) / 2) * target_w
        h_in_model = ((h_norm + 1) / 2) * target_h
        
        # 3. Scale back to Original Image Space
        # We divide by 'scale' to undo the resizing.
        # Note: This assumes Top-Left padding (0,0). 
        # If you used Centered padding, you would subtract the padding offset here first.
        x = x_in_model / scale
        y = y_in_model / scale
        w = w_in_model / scale
        h = h_in_model / scale
        
        # 4. Convert Center-Format (x,y) to Top-Left-Format (x1,y1,x2,y2)
        # The model predicts the center of the box. PIL needs corners.
        x1 = x - (w / 2)
        y1 = y - (h / 2)
        x2 = x + (w / 2)
        y2 = y + (h / 2)
        box_coords = [float(x1), float(y1), float(x2), float(y2)]
        processed_boxes.append(box_coords)
    # Prepare the dictionary structure
    data = {
        'pred_box': processed_boxes
    }
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully saved box to {save_path}")

def calculate_iou(box1, box2):
    # (Same IoU function as before)
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    xi1, yi1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    xi2, yi2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    return inter_area / (union_area + 1e-6)

def get_args():
    parser = argparse.ArgumentParser()
    # Path to the specific checkpoint you want to test
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    # Optional override for data path
    parser.add_argument("--test_data", type=str, default=None, help="Override path to test data")
    return parser.parse_args()

@torch.no_grad()
def test():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Output Setup # Diff-Location/samples_cocount/processed_dataset
    # output_dir = "./samples_real/Real_difficult/output_multiple_sampling"
    output_dir = "./samples_cocount/processed_dataset/output_multiple_sampling_density1class_inf100"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Configs
    # We load standard configs first, then override
    # model_cfg = yaml.safe_load(open("config/model_config.yaml", 'r'))
    # train_cfg = load_config("config/default.yaml")
    model_cfg = yaml.safe_load(open("checkpoints/best_ckpt/model_config_final.yaml", 'r'))
    train_cfg = load_config("checkpoints/best_ckpt/train_config_final.yaml")
    
    # 3. SMART CHECKPOINT LOADING
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading checkpoint from {args.checkpoint}...")
    # Load to CPU first to inspect metadata
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # --- FIX START: Detect num_steps from checkpoint ---
    if 'args' in checkpoint and 'num_steps' in checkpoint['args']:
        trained_steps = checkpoint['args']['num_steps']
        if trained_steps is not None:
            print(f"Detected training steps from checkpoint: {trained_steps}")
            # Inject into model config so buffers initialize correctly
            if 'diffusion' not in model_cfg: model_cfg['diffusion'] = {}
            model_cfg['diffusion']['num_timesteps'] = trained_steps
            # Also helpful to set a variable for the sampling loop later
            num_sampling_steps = 50 # Or use trained_steps for full quality
    else:
        print("Warning: Could not detect num_steps in checkpoint. Assuming default (100).")
        # If your checkpoint failed here, manually set:
        # model_cfg['diffusion'] = {'num_timesteps': 1000} 
    # --- FIX END ---

    # 4. Initialize Model (Now with correct shape)
    model = ObjectPlacementPolicy(model_cfg)
    
    # 5. Load Weights
    # Now this will work because shapes match
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 6. Dataset
    test_path = args.test_data if args.test_data else train_cfg['training']['data'].get('test_path', 'data/test')
    print(f"Testing data from: {test_path}")
    dataset = ObjectPlacementDataset(test_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    ious = []
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        print(f"Processing sample {i+1}/{len(dataloader)}")
        rgb = batch['pixel_values'].to(device)
        density = batch['density_map'].to(device)
        gt_box = batch['bbox'].numpy()[0]
        text = batch['text']
        scale = batch['scale'].item()
        
        # Inference
        vis_emb = model.vision_encoder(rgb, density)
        text_emb = model.text_encoder(text)
        global_cond = torch.cat([vis_emb, text_emb], dim=-1).expand(30, -1)
        
        box = torch.randn((30, 4), device=device)
        # Sampling Loop
        # We use a faster schedule (e.g., 50 steps) for testing even if trained on 1000
        inference_steps = 100 #50 
        
        for t in reversed(range(inference_steps)):
            t_batch = torch.full((30,), t, device=device, dtype=torch.long)
            box_in = box.unsqueeze(1) 
            
            # Note: We must be careful if the model expects t up to 1000 but we only loop 50.
            # Ideally, we map t (0-50) -> t_model (0-1000).
            # For simple linear testing, we can just use the model's full steps:
            # t_model = int(t * (model.num_timesteps / inference_steps))
            # t_batch = torch.full((1,), t_model, device=device, dtype=torch.long)
            # import pdb; pdb.set_trace()
            noise_pred = model.noise_net(box_in, t_batch, global_cond=global_cond).squeeze(1)
            box = box - (1.0/inference_steps) * noise_pred
            
        # Convert all 30 generated boxes to numpy: shape (30, 4)
        pred_boxes = box.cpu().numpy() 
        
        # --- MODIFICATION: Handle IoU for multiple boxes ---
        # Assuming `calculate_iou` expects a 1D array of 4 coordinates, 
        # we calculate IoU for all 30 and record the best matching box.
        best_iou = max([calculate_iou(b, gt_box) for b in pred_boxes])
        ious.append(best_iou)
        
        # Save visualization
        filename = dataset.files[i]
        # save_path = os.path.join(output_dir, f"result_{filename}")
        # visualize_result(
        #     image_path=os.path.join(dataset.image_dir, filename),
        #     pred_box=pred_box,
        #     scale=scale,
        #     save_path=save_path,
        #     # original_size=(512, 512)
        # )
        # Save JSON with box coordinates
        save_path = os.path.join(output_dir, f"{filename.split('.')[0]}.json")
        save_pred_box(pred_boxes, scale, save_path)
        # import pdb; pdb.set_trace()

    print(f"Mean IoU: {np.mean(ious):.4f}")

if __name__ == "__main__":
    test()