import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Téléchargement automatique du modèle HumanSeg si besoin
import os
import urllib.request

HUMANSEG_WEIGHTS = "models/human_seg_fp16.onnx"
HUMANSEG_URL = "https://huggingface.co/onnx/models/resolve/main/human-segmentation/human-seg_fp16.onnx"

def download_humanseg():
    os.makedirs(os.path.dirname(HUMANSEG_WEIGHTS), exist_ok=True)
    if not os.path.exists(HUMANSEG_WEIGHTS):
        print("Téléchargement du modèle HumanSeg ONNX...")
        urllib.request.urlretrieve(HUMANSEG_URL, HUMANSEG_WEIGHTS)
        print("Modèle HumanSeg téléchargé.")

def extract_silhouette(image_path, output_mask_path):
    # 1. Téléchargement auto HumanSeg si absent
    download_humanseg()

    # 2. Chargement image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((192, 192))

    # 3. Prétraitement pour HumanSeg
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)[None, ...]  # B, C, H, W

    # 4. Inference ONNX
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(HUMANSEG_WEIGHTS, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: img_np}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0][0, 0]  # (192, 192)
    mask = (logits > 0.5).astype(np.uint8) * 255

    # 5. Upscale au format d'origine + morph closing pour lisser
    mask_up = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((7,7), np.uint8)
    mask_closed = cv2.morphologyEx(mask_up, cv2.MORPH_CLOSE, kernel)

    # 6. Extraction du plus grand contour pour éviter les artefacts
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_closed)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_final, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        mask_final = mask_closed

    cv2.imwrite(output_mask_path, mask_final)
    print(f"[OK] Masque silhouette haute précision généré : {output_mask_path}")