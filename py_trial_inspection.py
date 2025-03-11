from dataset import CADDataLoader
from config import config, update_config
import argparse
import logging
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

args = argparse.Namespace()
args.cfg = "config/hrnet48.yaml"
args.val_only = True
args.test_only = False
args.data_root = "./data/floorplancad_v2"
args.embed_backbone = "hrnet48"
args.pretrained_model = "./pretrained_models/HRNet_W48_C_ssld_pretrained.pth"
args.local_rank = 0
args.log_step = 100
args.img_size = 700
args.max_prim = 12000
args.load_ckpt = ""
args.resume_ckpt = ""
args.log_dir = ""
args.seed = 304
args.debug = False
args.visualize = False
args.epoch = 1
args.opts = ""

cfg = update_config(config, args)
val_dataset = CADDataLoader(split="val", do_norm=cfg.do_norm, cfg=cfg)

image, xy, target, rgb_info, nns, offset, instance, indexes, basename = val_dataset[2]
print(basename)
var = image
print(var.shape)
print(torch.unique(var))



import numpy as np
ann_path = "processed/npy/val/0001-0001.npy"
npy_data = np.load(ann_path, allow_pickle=True).item()

var = "cat"
print(len(npy_data[var]))
print(npy_data[var][1000:1000+5])


from PIL import Image, ImageDraw

# Example list of (x, y) coordinates
coordinates = [(10, 10), (50, 50), (100, 100), (150, 150)]

# Define image size (adjust as needed)
width = max(x for x, y in coordinates) + 20
height = max(y for x, y in coordinates) + 20

# Create a white background image
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Draw dots
dot_radius = 3
for x, y in coordinates:
    draw.ellipse((x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius), fill="black")

# Save the image
image.save("output.png")

# Show the image (optional)
image.show()

