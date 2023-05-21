from PIL import Image
import requests
from rembg import remove
import sys
def paste_foreground_on_background(foreground_image, background_path, output_path):
    
    foreground_image = Image.open(foreground_image)

    
    if foreground_image.mode != "RGBA":
        foreground_image = foreground_image.convert("RGBA")

    
    with remove(foreground_image) as alpha_mask:
      
        background_image = Image.open(background_path)

        
        foreground_image = foreground_image.resize(background_image.size, Image.ANTIALIAS)
        
        if foreground_image.mode != background_image.mode:
            foreground_image = foreground_image.convert(background_image.mode)
        
        alpha_mask=alpha_mask.resize(background_image.size,Image.ANTIALIAS)

        background_image.paste(foreground_image, (0, 0),mask=alpha_mask)

        
        background_image.save(output_path)

foreground_image = sys.argv[1]
background_path = sys.argv[2]
output_path = "output.jpg"

paste_foreground_on_background(foreground_image, background_path, output_path)
