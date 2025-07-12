import sys
import os
from pipelines.convertor import detection, estimation, uv_mapping, fusion

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

def process_image(image_path):
    detected = detection.run(image_path)
    estimated = estimation.run(detected)
    uv_texture = uv_mapping.run(estimated)
    final_texture = fusion.run(uv_texture)
    model_id = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join("models", f"{model_id}_texture.png")
    os.makedirs("models", exist_ok=True)
    final_texture.save(output_path)
    print(f"Texture générée : {output_path}")

def main():
    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"Le dossier '{input_dir}' n'existe pas. Merci de le créer et d'y placer vos images.")
        sys.exit(1)
    images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not images:
        print(f"Aucune image trouvée dans '{input_dir}'.")
        sys.exit(1)
    for image_path in images:
        print(f"\n---\nTraitement de l'image : {image_path}")
        process_image(image_path)

if __name__ == "__main__":
    main()