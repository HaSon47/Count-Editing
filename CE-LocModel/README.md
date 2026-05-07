# CE-LocModel

A diffusion-based object localization model that predicts bounding boxes conditioned on an RGB image, a density map, and a text class label. It uses a DDPM-style reverse diffusion process with a ResNet18 vision encoder, a frozen CLIP text encoder, and a conditional 1D U-Net noise prediction network.

---

## Environment Setup

### Requirements
- Python 3.10
- CUDA 12.1 (for GPU acceleration)
- Conda (recommended)

### Create and activate environment

```bash
conda create -n ce-locmodel python=3.10 -y
conda activate ce-locmodel
```

### Install PyTorch with CUDA 12.1

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

### Install remaining dependencies

```bash
pip install -r requirements.txt
```

> Note: the `torch` and `torchvision` lines in `requirements.txt` serve as version documentation. Install them via the command above to get the correct CUDA variant.

---

## Download Model and Data

All required files are hosted in this shared folder:

**https://adelaideuniversity.box.com/s/cxrzvy0s33llavj7n5nfzzp4gspz8os1**

The folder contains two files:
- `best_model.pth` — pretrained model weights (~435 MB)
- `data.zip` — dataset (train and test splits)

### Step 1 — Download both files

Open the link in a browser and download `best_model.pth` and `data.zip` manually, or use the Box direct-download URLs:

```bash
curl -L "https://adelaideuniversity.box.com/shared/static/best_model.pth" -o best_model.pth
curl -L "https://adelaideuniversity.box.com/shared/static/data.zip"       -o data.zip
```

### Step 2 — Place the model checkpoint

```bash
mkdir -p checkpoints/best_ckpt
mv best_model.pth checkpoints/best_ckpt/
```

Expected layout:

```
checkpoints/best_ckpt/
├── best_model.pth          # Model weights (~435 MB)
├── model_config_final.yaml # Model architecture config (already in repo)
└── train_config_final.yaml # Training hyperparameters (already in repo)
```

### Step 3 — Extract and place the dataset

```bash
unzip data.zip
mv data/train samples/train
mv data/test  samples/test
```

Expected layout after extraction:

```
samples/
├── train/
│   ├── images/       # RGB images (*.jpg, *.png)
│   ├── density/      # Grayscale density maps (*.png, same stem as images)
│   └── annotation/   # JSON annotations (*.json, same stem as images)
└── test/
    ├── images/
    ├── density/
    └── annotation/
```

### Annotation format

Each JSON file in `annotation/` corresponds to one image:

```json
{
  "class": "object class name",
  "target_bbox": [center_x, center_y, width, height]
}
```

Coordinates are in absolute pixels of the original image.

---

## Training

Edit `config/default.yaml` to set hyperparameters (batch size, learning rate, epochs, data paths), then run:

```bash
bash train.sh
```

The script calls:

```bash
CUDA_VISIBLE_DEVICES=1 python train_w_args.py --epochs 200 --batch_size 32 --lr 5e-5
```

Checkpoints are saved to `checkpoints/` after each epoch. The best model (lowest validation loss) is saved to `checkpoints/best_ckpt/best_model.pth`.

### Key training config (`config/default.yaml`)

| Parameter | Default | Final trained value |
|---|---|---|
| `batch_size` | 32 | 32 |
| `learning_rate` | 1e-4 | 5e-5 |
| `num_epochs` | 100 | 300 |
| `num_timesteps` | 100 | 1000 |
| `train_path` | `samples/train/` | `samples/train/` |

---

## Testing

```bash
bash test.sh
```

The script calls:

```bash
CUDA_VISIBLE_DEVICES=0 python test_mul_box.py --checkpoint checkpoints/best_ckpt/best_model.pth
```

This runs multiple-sampling inference (30 bounding box samples per image) and selects the prediction with the highest IoU against the ground truth. Results are written to:

```
samples_cocount/processed_dataset/output_multiple_sampling_density1class_inf100/
```

Each output file contains the predicted bounding box and visualization image. Mean IoU is printed to stdout when evaluation completes.

---

## Inference on a single image

Use `inference.py` for a quick single-image test:

```python
python inference.py
```

Edit the file to set your image path, density map path, and text prompt. Output images with drawn bounding boxes are saved to `examples/outputs/`.

---

## Model Architecture

```
ObjectPlacementPolicy
├── SpatialVisualEncoder       ResNet18 (4-channel input: RGB + density)
│                              → SpatialSoftmax → Linear → 128D
├── CLIPTextEncoder            openai/clip-vit-base-patch32 (frozen)
│                              → Linear + Mish → 128D
└── ConditionalUnet1D          1D U-Net, input [B,1,4] (cx,cy,w,h)
                               conditioned on 256D (vision + text)
                               → predicts noise ε at each timestep
```

Diffusion schedule: linear DDPM, β from 0.0001 to 0.02 over 1000 steps.
Bounding boxes are normalized to `[-1, 1]` during training and denormalized at inference.
