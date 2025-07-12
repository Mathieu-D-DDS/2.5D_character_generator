import os

from pipelines.convertor_v2.segmentation import extract_silhouette
from pipelines.convertor_v2.texture_front import generate_front_texture
from pipelines.convertor_v2.mirror_and_back import generate_back_texture
from pipelines.convertor_v2.side_projection import generate_side_masks
from pipelines.convertor_v2.texture_side import generate_side_texture
from pipelines.convertor_v2.uv_assembly import assemble_uv

# Chemins fichiers
INPUT_IMAGE = "input/model_alpha.jpeg"
MASK_FRONT = "models/model_alpha_mask.png"
TEXTURE_FRONT = "models/model_alpha_texture_front.png"
MASK_BACK = "models/model_alpha_mask_back.png"
TEXTURE_BACK = "models/model_alpha_texture_back.png"
MASK_LEFT = "models/model_alpha_mask_left.png"
MASK_RIGHT = "models/model_alpha_mask_right.png"
TEXTURE_LEFT = "models/model_alpha_texture_left.png"
TEXTURE_RIGHT = "models/model_alpha_texture_right.png"
UV_MAP = "models/model_alpha_uv_map.png"

def main():
    # 1. Segmentation frontale
    extract_silhouette(INPUT_IMAGE, MASK_FRONT)
    # 2. Génération texture frontale
    generate_front_texture(INPUT_IMAGE, MASK_FRONT, TEXTURE_FRONT)
    # 3. Mirroring + génération dos
    generate_back_texture(INPUT_IMAGE, MASK_FRONT, TEXTURE_BACK, MASK_BACK)
    # 4. Génération masques latéraux
    generate_side_masks(MASK_FRONT, MASK_BACK, MASK_LEFT, MASK_RIGHT)
    # 5. Génération textures profils
    generate_side_texture(INPUT_IMAGE, MASK_LEFT, TEXTURE_LEFT, "left")
    generate_side_texture(INPUT_IMAGE, MASK_RIGHT, TEXTURE_RIGHT, "right")
    # 6. Assemblage UV
    assemble_uv(TEXTURE_FRONT, TEXTURE_BACK, TEXTURE_LEFT, TEXTURE_RIGHT, UV_MAP)
    print(f"Texture UV générée : {UV_MAP}")

if __name__ == "__main__":
    main()