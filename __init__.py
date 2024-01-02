
from rembg import remove, new_session
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
                "transparency": ("BOOLEAN", {"default": True},),
                "model": (["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use", "isnet-anime"],),
                "post_processing": ("BOOLEAN", {"default": False}),
                "only_mask": ("BOOLEAN", {"default": False},),
                "alpha_matting": ("BOOLEAN", {"default": False},),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "alpha_matting_erode_size": ("INT", {"default": 10, "min": 0, "max": 255}),
                "background_color": (["none", "black", "white", "magenta", "chroma green", "chroma blue"],),
                # "putalpha": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_rembg"

     # A helper function to convert from strings to logical boolean
     # Conforms to https://docs.python.org/3/library/stdtypes.html#truth-value-testing
     # With the addition of evaluating string representations of Falsey types
    def __convertToBool(self, x):

        # Evaluate string representation of False types
        if type(x) == str:
            x = x.strip()
            if (x.lower() == 'false'
                or x.lower() == 'none'
                or x == '0'
                or x == '0.0'
                or x == '0j'
                or x == "''"
                or x == '""'
                or x == "()"
                or x == "[]"
                or x == "{}"
                or x.lower() == "decimal(0)"
                or x.lower() == "fraction(0,1)"
                or x.lower() == "set()"
                or x.lower() == "range(0)"
            ):
                return False
            else:
                return True

        # Anything else will be evaluated by the bool function
        return bool(x)

    def tensor_to_pil(self, img):
        if img is not None:
            i = 255. * img.cpu().numpy().squeeze()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    
    def image_rembg(
            self,
            images,
            transparency=True,
            model="u2net",
            alpha_matting=False,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_processing=False,
            only_mask=False,
            background_color="none",
            # putalpha = False,
    ):

        # ComfyUI will allow strings in place of booleans, validate the input.
        transparency = transparency if type(transparency) is bool else self.__convertToBool(transparency)
        alpha_matting = alpha_matting if type(alpha_matting) is bool else self.__convertToBool(alpha_matting)
        post_processing = post_processing if type(post_processing) is bool else self.__convertToBool(post_processing)
        only_mask = only_mask if type(only_mask) is bool else self.__convertToBool(only_mask)

        # Set bgcolor
        bgrgba = None
        if background_color == "black":
            bgrgba = [0, 0, 0, 255]
        elif background_color == "white":
            bgrgba = [255, 255, 255, 255]
        elif background_color == "magenta":
            bgrgba = [255, 0, 255, 255]
        elif background_color == "chroma green":
            bgrgba = [0, 177, 64, 255]
        elif background_color == "chroma blue":
            bgrgba = [0, 71, 187, 255]
        else:
            bgrgba = None

        if transparency and bgrgba is not None:
            bgrgba[3] = 0
        
        total_images = []
        for image in images:
            image = self.tensor_to_pil(image)
            image = remove(
                    image,
                    session=new_session(model),
                    # post_process_mask=post_processing,
                    # alpha_matting=alpha_matting,
                    # alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    # alpha_matting_background_threshold=alpha_matting_background_threshold,
                    # alpha_matting_erode_size=alpha_matting_erode_size,
                    # only_mask=only_mask,
                    bgcolor=bgrgba,
                    # putalpha = putalpha,
                )

            # Convert to tensor
            total_images.append(torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0))

        total_images = torch.cat(total_images, 0)
        return (total_images,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Video Remove Background (rembg)": ImageRemoveBackgroundRembg
}

