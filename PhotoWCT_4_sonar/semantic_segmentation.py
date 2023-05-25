from io import BytesIO
from PIL import Image
import requests
from rembg import remove
import rembg
import numpy as np

import sys
    
def paste_foreground_on_background(foreground_image, background_path, output_path=None):
    
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

        
        if output_path:
            background_image.save(output_path)
        else:
            return background_image



def separate_foreground_background(image_path):

    with open(image_path, "rb") as f:
        image_data = f.read()

    
    output = rembg.remove(image_data)
    
    # Convert the output data to a numpy array
    og_image=np.frombuffer(image_data,dtype=np.uint8)
    output_data = np.frombuffer(output, dtype=np.uint8)
    # Decode the image using PIL
    img = Image.open(BytesIO(output_data))
    og_image=Image.open(BytesIO(og_image))
    

    # Split the image into foreground and background
    alpha_channel = np.array(img.split()[-1])  # Alpha channel
    alpha_channel_inv=np.array(img.split()[-1].point(lambda p:255-p))
    # img.split()[-1].point(lambda p:0)
    main_image = og_image.convert("RGB")  # RGB channels
    main_image.save("temp_image.jpg")
    # Apply the alpha channel as a mask to extract the foreground

    # print(np.expand_dims(alpha_channel, axis=2))
    # print(np.invert(np.expand_dims(alpha_channel,axis=2)))
    foreground = Image.fromarray(np.array(main_image) * np.expand_dims(alpha_channel, axis=2))
    background=Image.fromarray(np.array(main_image) * np.expand_dims(alpha_channel_inv, axis=2))
    # Invert the alpha channel to get the background mask
    # background_mask = np.invert(alpha_channel)

    # # Create a white background image of the same size as the foreground
    # background = Image.new("RGB", foreground.size, (255, 255, 255))

    # # Apply the background mask to extract the background
    # background = Image.fromarray(np.array(background) * np.expand_dims(background_mask, axis=2))

    return foreground, background
    



