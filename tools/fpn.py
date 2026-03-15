# FPN
# extracts FPN feature maps

import sys
import os
from xml.parsers.expat import model
import torch
import cv2
import numpy as np
from yolox.exp import get_exp

def load_aabb_from_txt(txt_path, img_width, img_height):
    """
    Reads a YOLO-style AABB annotation and converts it to pixel coordinates.

    Args:
        txt_path (str): path to .txt annotation
        img_width (int): image width in pixels
        img_height (int): image height in pixels

    Returns:
        tuple: (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    with open(txt_path, "r") as f:
        line = f.readline().strip()  # read the first line
        if len(line) == 0:
            raise ValueError(f"No annotation in {txt_path}")
        parts = line.split()
        cls, xc, yc, w, h = map(float, parts)

        x_min = int((xc - w/2) * img_width)
        y_min = int((yc - h/2) * img_height)
        x_max = int((xc + w/2) * img_width)
        y_max = int((yc + h/2) * img_height)

        # clamp
        x_min = max(0, min(img_width, x_min))
        x_max = max(0, min(img_width, x_max))
        y_min = max(0, min(img_height, y_min))
        y_max = max(0, min(img_height, y_max))

        return (x_min, y_min, x_max, y_max)
    
# map ROI to feature maps using stride
strides = {"p3": 8, "p4": 16, "p5": 32}
    
def roi_on_fmap(bbox, fmap_shape, stride):
    x_min, y_min, x_max, y_max = bbox
    Hf, Wf = fmap_shape[-2], fmap_shape[-1]
    x1 = int(x_min / stride)
    y1 = int(y_min / stride)
    x2 = int(np.ceil(x_max / stride))
    y2 = int(np.ceil(y_max / stride))
    x1 = max(0, min(Wf, x1))
    x2 = max(0, min(Wf, x2))
    y1 = max(0, min(Hf, y1))
    y2 = max(0, min(Hf, y2))
    return x1, y1, x2, y2

def get_pN_vector(img_path, txt_path, model, N=3):
    """Load image + annotation, forward pass, return pooled P3 vector"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    img_tensor = cv2.resize(img, (640,640)).transpose(2,0,1)
    img_tensor = torch.from_numpy(img_tensor).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # batch

    bbox = load_aabb_from_txt(txt_path, W, H)

    with torch.no_grad():
        p3, p4, p5 = model.backbone(img_tensor)
        p = {"p3": p3, "p4": p4, "p5": p5}[f"p{N}"]
        x1, y1, x2, y2 = roi_on_fmap(bbox, p.shape, stride=strides[f"p{N}"])
        roi = p[:, :, y1:y2, x1:x2]
        vec = torch.mean(roi, dim=(2,3))  # [1, C]
        vec = vec / vec.norm(p=2, dim=1, keepdim=True)  # normalize
    return vec

# load experiment configuration
exp = get_exp("../exps/default/yolox_s", None)
model = exp.get_model()
model.eval()

# load weights 
ckpt = torch.load("../yolox_s.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

# load image
img_dir = "../../yolox-oneshot/datasets/COCO3/train2017/"
txt_dir = "../../yolox-oneshot/datasets/COCO3/annotations/train2017/"

# N = scale 3, 4, 5
N = 5

# reference
rfilename = "candy.200.01_640x640.png"
ref_vec = get_pN_vector(os.path.join(img_dir, rfilename), os.path.join(txt_dir, rfilename.replace(".png", ".txt")), model, N=N)

# query
qfilename = "candy.200.01_640x640.png"
qfilename = "candy.300.03_640x640.png"
qfilename = "candy.300.14_640x640.png"
qfilename = "cards.200.04_640x640.png"
qfilename = "three.600.53_640x640.png"
query_vec = get_pN_vector(os.path.join(img_dir, qfilename), os.path.join(txt_dir, qfilename.replace(".png", ".txt")), model, N=N)

# cosine similarity
cos = torch.nn.CosineSimilarity(dim=1)
sim = cos(ref_vec, query_vec)
print(f"Cosine similarity between {rfilename} and {qfilename} P3={sim.item():.2f}")

"""
        Filename              P5 Similarity
Ref   = candy.200.01_640x640
Query = candy.200.01_640x640  1.00
      = candy.300.03_640x640  0.93
      = candy.300.14_640x640  0.95
      = cards.200.04_640x640  0.88
      = three.600.53_640x640  0.89
"""