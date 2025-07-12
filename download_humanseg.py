import os
import urllib.request

HUMANSEG_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/ultrafast_human_segmentation/model/ultrafast_human_segmentation.onnx"
DEST = os.path.join("models", "ultrafast_human_segmentation.onnx")

os.makedirs("models", exist_ok=True)
if not os.path.exists(DEST):
    print(f"Téléchargement du modèle HumanSeg ONNX...")
    urllib.request.urlretrieve(HUMANSEG_URL, DEST)
    print(f"→ Modèle téléchargé dans {DEST}")
else:
    print(f"[OK] Modèle HumanSeg déjà présent dans {DEST}")