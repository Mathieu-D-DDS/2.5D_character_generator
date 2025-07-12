import sys
import os
import torch
import cv2
import numpy as np

# Ajoute le dossier racine pour que 'midas' soit reconnu comme package
sys.path.insert(0, os.path.abspath("."))

from midas.dpt_depth import DPTDepthModel

# Choisis ici le backbone et le poids (Tiny ou Base, les deux sont compatibles)
BACKBONE = "swin2t16_256"  # Pour Tiny
MODEL_PATH = os.path.join("midas", "dpt_swin2_tiny_256.pt")

# Pour Base (si tu veux essayer, décommente ces deux lignes)
# BACKBONE = "swin2b24_384"
# MODEL_PATH = os.path.join("midas", "dpt_swin2_base_384.pt")

def run(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPTDepthModel(
        path=MODEL_PATH,
        backbone=BACKBONE,
        non_negative=True,
        enable_attention_hooks=False,
    )
    model.eval()
    model.to(device)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
        depth = prediction.squeeze().cpu().numpy()
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
    return depth_norm