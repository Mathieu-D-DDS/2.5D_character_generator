from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import json
import os

def load_prompts(json_path="prompts.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_side_texture(image_path, mask_side_path, output_path, side):
    prompts = load_prompts()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    prompt = prompts.get(side, f"{side} side profile view of a person")
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_side_path).convert("L")
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    result.save(output_path)