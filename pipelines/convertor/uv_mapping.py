def run(image_path):
    # TODO: Générer la texture à plat (projection UV + inpainting)
    print(f"[UV Mapping] Génération de la texture UV pour {image_path}")
    from PIL import Image
    # Placeholder: retourne une image unie pour test
    return Image.new("RGBA", (1024, 1024), (128, 128, 128, 255))