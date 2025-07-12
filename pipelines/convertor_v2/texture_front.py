from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

def generate_front_texture(image_path, mask_path, output_path):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None  # Désactive le filtre NSFW
    ).to("cuda")
    prompt = "front view of a person wearing clothes, realistic, full body, natural lighting, high detail"
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    result.save(output_path)