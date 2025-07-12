import sys
import os
import torch
import cv2
import numpy as np

# Ajoute le dossier racine pour que 'midas' soit reconnu comme package
sys.path.insert(0, os.path.abspath("."))

from midas.dpt_depth import DPTDepthModel

def run(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("midas", "dpt_swin2_tiny_256.pt")
    model = DPTDepthModel(
        path=model_path,
        backbone="swin2t16_256",
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