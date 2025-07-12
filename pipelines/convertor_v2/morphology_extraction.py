import json
import cv2
import numpy as np
import os

def load_landmarks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def landmark_coords_px(landmarks, w, h):
    coords = {}
    for k, v in landmarks.items():
        coords[k] = [int(v["x"] * w), int(v["y"] * h)]
    return coords

def segment_length(a, b):
    return int(np.linalg.norm(np.array(a) - np.array(b)))

def extract_width_at(mask, y):
    # Récupère la largeur du mask à une hauteur y (ligne horizontale)
    row = mask[int(y), :]
    xs = np.where(row > 0)[0]
    if len(xs) == 0:
        return 0
    return int(xs[-1] - xs[0])

def run(skeleton_json, mask_path, output_json):
    data = load_landmarks(skeleton_json)
    w, h = data["image_width"], data["image_height"]
    lmk_px = landmark_coords_px(data["keypoints"], w, h)

    # Charge le mask de segmentation (uint8, 0/255)
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print("Mask non trouvé !")
        return

    # Exemples de segments à mesurer (adaptable)
    segments = {
        "left_upper_arm": ("landmark_11", "landmark_13"),
        "left_forearm": ("landmark_13", "landmark_15"),
        "right_upper_arm": ("landmark_12", "landmark_14"),
        "right_forearm": ("landmark_14", "landmark_16"),
        "left_thigh": ("landmark_23", "landmark_25"),
        "left_calf": ("landmark_25", "landmark_27"),
        "right_thigh": ("landmark_24", "landmark_26"),
        "right_calf": ("landmark_26", "landmark_28"),
        "shoulders": ("landmark_11", "landmark_12"),
        "hips": ("landmark_23", "landmark_24"),
        "spine": ("landmark_0", "landmark_24"), # ou moyenne Tête/Bassin
        "height": ("landmark_0", "landmark_32"),
    }

    segments_length = {}
    segments_width = {}

    for name, (l1, l2) in segments.items():
        if l1 in lmk_px and l2 in lmk_px:
            segments_length[name] = segment_length(lmk_px[l1], lmk_px[l2])
        else:
            segments_length[name] = None

    # Largeur des membres à la hauteur du premier landmark du segment
    for name, (l1, l2) in segments.items():
        if l1 in lmk_px:
            y = lmk_px[l1][1]
            width = extract_width_at(mask, y)
            segments_width[name] = width
        else:
            segments_width[name] = None

    # Compilation
    morpho = {
        "image_path": data["image_path"],
        "image_width": w,
        "image_height": h,
        "landmarks_px": lmk_px,
        "segments_length_px": segments_length,
        "segments_width_px": segments_width,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(morpho, f, indent=2)
    print(f"[OK] Morphologie extraite et sauvegardée dans : {output_json}")