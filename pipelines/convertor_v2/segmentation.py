import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import os

SAM_CHECKPOINT = "SAM/sam_vit_h_4b8939.pth"

def extract_silhouette(image_path, output_mask_path):
    # 1. Vérifie la présence du checkpoint
    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint SAM non trouvé : {SAM_CHECKPOINT}\n"
            "Télécharge-le ici : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )

    # 2. Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image non trouvée : {image_path}")

    # 3. Instancie le modèle SAM
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # 4. Prédiction du mask le plus grand (corps)
    # Utilise tout le cadre comme "prompt" pour obtenir le mask principal
    h, w = image.shape[:2]
    input_box = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False
    )
    mask = masks[0].astype(np.uint8) * 255

    # 5. Sauvegarde du mask
    cv2.imwrite(output_mask_path, mask)
    print(f"[OK] Masque silhouette ultra-précis généré : {output_mask_path}")