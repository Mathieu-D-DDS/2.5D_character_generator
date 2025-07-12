import os
from pipelines.convertor_v2.segmentation import extract_silhouette
from pipelines.convertor_v2 import skeleton_extraction, morphology_extraction

INPUT_IMAGE = "input/model_alpha.jpeg"
MASK_FRONT = "models/model_alpha_mask.png"
SKELETON_JSON = "models/model_alpha_skeleton.json"
MORPHO_JSON = "models/model_alpha_morphology.json"

def main():
    # 1. Générer le mask s'il n'existe pas
    if not os.path.exists(MASK_FRONT):
        extract_silhouette(INPUT_IMAGE, MASK_FRONT)
        print(f"Mask généré : {MASK_FRONT}")
    else:
        print(f"Mask déjà présent : {MASK_FRONT}")
    # 2. Extraire le squelette
    skeleton_extraction.extract_pose_landmarks(INPUT_IMAGE, SKELETON_JSON)
    # 3. Extraction morphologique
    morphology_extraction.run(SKELETON_JSON, MASK_FRONT, MORPHO_JSON)

if __name__ == "__main__":
    main()