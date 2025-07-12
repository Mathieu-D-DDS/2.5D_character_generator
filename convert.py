import sys
import os
from pipelines.convertor import detection, estimation, uv_mapping, fusion

def main(image_path):
    # 1. Détection / Segmentation
    detected = detection.run(image_path)
    # 2. Estimation pose/profondeur
    estimated = estimation.run(detected)
    # 3. UV Mapping + inpainting
    uv_texture = uv_mapping.run(estimated)
    # 4. Fusion/Post-process
    final_texture = fusion.run(uv_texture)
    # Sauvegarde
    model_id = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join("models", f"{model_id}_texture.png")
    os.makedirs("models", exist_ok=True)
    final_texture.save(output_path)
    print(f"Texture générée : {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])