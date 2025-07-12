import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def run(depth_map):
    # Pour le mockup, on crée un masque "UV" simple basé sur la profondeur
    mask = (depth_map > np.percentile(depth_map, 5)).astype(np.uint8) * 255
    uv_base = Image.fromarray(mask).convert("RGB").resize((512, 512))

    # Chargement du pipeline d'inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    pipe = pipe.to("cuda" if pipe.device.type == "cuda" else "cpu")

    # Prompt d'inpainting pour générer des zones non vues
    prompt = "full body skin texture flat, UV layout, realistic, seamless"
    inpaint_result = pipe(prompt=prompt, image=uv_base, mask_image=uv_base).images[0]
    return inpaint_result