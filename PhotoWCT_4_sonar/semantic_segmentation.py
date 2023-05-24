import os
from PIL import Image
import requests
from rembg import remove
import sys


def paste_foreground_on_background(foreground_image, background_path, output_name, output_path=None):

    if './' in str(foreground_image):
        foreground_image = Image.open(foreground_image)
    if foreground_image.mode != "RGBA":
        foreground_image = foreground_image.convert("RGBA")

    with remove(foreground_image) as alpha_mask:

        background_image = Image.open(background_path)

        foreground_image = foreground_image.resize(
            background_image.size, Image.ANTIALIAS)

        if foreground_image.mode != background_image.mode:
            foreground_image = foreground_image.convert(background_image.mode)

        alpha_mask = alpha_mask.resize(background_image.size, Image.ANTIALIAS)
        alpha_mask = alpha_mask.convert("L")
        mask_path = output_name.split('.jpg')
        mask_path = mask_path[0].replace('/output', '/output/segments')
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        alpha_mask.save(mask_path+'_segmented_mask.jpg')

        background_image.paste(foreground_image, (0, 0), mask=alpha_mask)
        alpha_mask.convert(background_image.mode)
        if output_path:
            background_image.save(output_path)
        else:
            return background_image, alpha_mask
