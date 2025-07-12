from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import cv2

def generate_back_texture(image_path, mask_front_path, output_texture_path, output_mask_path):
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
        safety_checker=None  # Désactive le filtre NSFW
    ).to("cuda")
    prompt = "back view of a person wearing clothes, realistic, full body, natural lighting, high detail"
    result = pipe(prompt=prompt, image=pil_image, mask_image=pil_mask).images[0]
    result.save(output_texture_path)