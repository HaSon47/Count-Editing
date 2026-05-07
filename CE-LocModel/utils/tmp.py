from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===== LOAD =====
img = np.array(Image.open("/mnt/disk2/hachi/data/Location/Image/2_1.png"))
density_png = np.array(
    Image.open("/mnt/disk2/hachi/data/Location/Density/2_1.png").convert("RGB") # IMPORTANT: với density image cần convert về 'RGB' vì mode hiện tại là 'RGBA' (H,W,4)
)
density_npy = np.load("/mnt/disk2/hachi/data/Location/Density_npy/2_1.npy")

anno_path = "/mnt/disk2/hachi/data/Location/Anno/2_1.json"
with open(anno_path, "r") as f:
    anno = json.load(f)

class_name = anno["class"]
target_bbox = anno["target_bbox"]  # xc, yc, w, h

# ===== BBOX PROCESS =====
h_img, w_img = img.shape[:2]

xc, yc, bw, bh = target_bbox
x_min = xc - bw / 2
y_min = yc - bh / 2

# ===== OVERLAY + DRAW BBOX =====
fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(img)
ax.imshow(density_png, alpha=0.3)
ax.imshow(density_npy, cmap='jet', alpha=0.3)

rect = patches.Rectangle(
    (x_min, y_min),
    bw,
    bh,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
)
ax.add_patch(rect)

ax.set_title(f"Overlay + BBox ({class_name})")
ax.axis("off")

plt.show()
plt.savefig('tmp.png')
plt.close()

print("Ok")