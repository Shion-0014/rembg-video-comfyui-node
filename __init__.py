
from rembg import remove
from PIL import Image
import torch
import numpy as np

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
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def tensor_to_pil(self, img):
        if img is not None:
            i = 255. * img.cpu().numpy().squeeze()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def add_green_background(self, image):
        # Create a green background
        green_background = Image.new("RGB", image.size, (0, 255, 0))

        # Overlay the transparent image on the green background
        green_background.paste(image, (0, 0), image)
        return green_background

    def remove_background(self, images):
        # Create empty tensor list
        total_images = []
        for image in images:
            image = self.tensor_to_pil(image)
            image = remove(image)

            # Add green background
            image_with_green_bg = self.add_green_background(image)

            # Convert to tensor
            total_images.append(torch.from_numpy(np.array(image_with_green_bg).astype(np.float32) / 255.0).unsqueeze(0))

        total_images = torch.cat(total_images, 0)
        return (total_images,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Video Remove Background (rembg)": ImageRemoveBackgroundRembg
}

