from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import cv2
import json
import os

def load_prompts(json_path="prompts.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_back_texture(image_path, mask_front_path, output_texture_path, output_mask_path):
    prompts = load_prompts()
    # Mirror mask
    mask_front = cv2.imread(mask_front_path, 0)
    mask_back = cv2.flip(mask_front, 1)
    cv2.imwrite(output_mask_path, mask_back)
    # Mirror image for conditioning
    image = cv2.imread(image_path)
    image_back = cv2.flip(image, 1)
    pil_image = Image.fromarray(cv2.cvtColor(image_back, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_back)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    prompt = prompts.get("back", "back view of a person")
    result = pipe(prompt=prompt, image=pil_image, mask_image=pil_mask).images[0]
    result.save(output_texture_path)