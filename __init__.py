
from rembg import remove
from PIL import Image
import torch
import numpy as np

# Tensor to PIL
def tensor2pil(image_tensor):
    image_np = image_tensor.cpu().numpy().squeeze()
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    if image_np.ndim == 2:  # Grayscale image
        return Image.fromarray(image_np, mode='L')
    return Image.fromarray(image_np)

# Convert PIL to Tensor
def pil2tensor(image_pil):
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    if image_np.ndim == 2:  # Grayscale image
        image_np = np.expand_dims(image_np, axis=2)
    return torch.from_numpy(image_np).unsqueeze(0)

class ImageRemoveBackgroundRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_video_background"
    CATEGORY = "image"

    def remove_background(self, images):
        processed_images = []
        for image_tensor in images:
            pil_image = tensor2pil(image_tensor)
            removed_bg_image = remove(pil_image)
            tensor_image = pil2tensor(removed_bg_image)
            processed_images.append(tensor_image)
        return tuple(processed_images,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image Remove Background (rembg)": ImageRemoveBackgroundRembg
}
