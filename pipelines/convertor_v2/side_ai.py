from PIL import Image, ImageChops
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch
import json

def generate_side_mask_ai(mask_front_path, mask_back_path, output_side_path, side="left", prompts_json="prompts.json"):
    # Charger les masques front et back
    front = Image.open(mask_front_path).convert("L")
    back = Image.open(mask_back_path).convert("L")
    # Fusionner les deux (overlay + binarisation)
    blend = ImageChops.add(front, back, scale=2.0).point(lambda x: 255 if x > 128 else 0)
    blend = blend.resize((512, 512))
    # Charger le prompt depuis JSON
    with open(prompts_json, "r") as f:
        prompts = json.load(f)
    prompt = prompts.get(side, f"{side} side profile mask of a person, full body, proportions matching front and back views, white silhouette on black background")
    # Pipeline SD inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    # Masque total (on demande à l'IA de tout imaginer)
    mask = Image.new("L", blend.size, 255)
    result = pipe(prompt=prompt, image=blend.convert("RGB"), mask_image=mask).images[0]
    # Binarisation du résultat
    arr = np.array(result.convert("L"))
    arr = np.where(arr > 128, 255, 0).astype(np.uint8)
    Image.fromarray(arr).save(output_side_path)