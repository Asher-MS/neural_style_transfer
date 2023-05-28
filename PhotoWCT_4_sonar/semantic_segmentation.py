from io import BytesIO
from PIL import Image,ImageEnhance,ImageDraw,ImageFilter
import requests
from rembg import remove
import rembg
import numpy as np
from noisy_edge import add_scale_noise
import sys

    
def paste_foreground_on_background(foreground_image, background_path, output_path=None,rotate_angle=0,add_noise=False):
    
    foreground_image = Image.open(foreground_image)

    
    if foreground_image.mode != "RGBA":
        foreground_image = foreground_image.convert("RGBA")

    
    with remove(foreground_image) as alpha_mask:
      
        background_image = Image.open(background_path)

        
        foreground_image = foreground_image.resize(background_image.size, Image.ANTIALIAS)
        
        if foreground_image.mode != background_image.mode:
            foreground_image = foreground_image.convert(background_image.mode)
        
        alpha_mask=alpha_mask.resize(background_image.size,Image.ANTIALIAS)
        if add_noise:
                foreground_image=add_scale_noise(image=foreground_image.convert("RGBA"),scale_factor=30,noise_intensity=1)
                alpha_mask=add_scale_noise(image=alpha_mask.convert("RGBA"),scale_factor=30,noise_intensity=1)


        foreground_image=foreground_image.rotate(rotate_angle,resample=0,expand=0)
        alpha_mask=alpha_mask.rotate(rotate_angle,resample=0,expand=0)
        shadow_added=add_shadow(foreground_image,alpha_mask,(0,0,0,127),127,10)
        shadow_added.show()
        foreground_image=shadow_added

        img_enhancer = ImageEnhance.Brightness(foreground_image)

        factor = 1
        foreground_image = img_enhancer.enhance(factor)
        background_image.paste(foreground_image, (0, 0),mask=alpha_mask)

        
        if output_path:
            background_image.save(output_path)
        else:
            return background_image,alpha_mask


def add_shadow(image, alpha_mask, shadow_color, shadow_opacity, blur_radius):

    # Open the image and alpha mask

    # Create a shadow image with the same size as the input image
    shadow = Image.new("RGBA", image.size)

    # Apply a Gaussian blur to the alpha mask
    alpha_mask_blur = alpha_mask.filter(ImageFilter.GaussianBlur(blur_radius))

    # Calculate the shadow offset based on the image size
    shadow_offset = (int(image.size[0] * 0.1), int(image.size[1] * 0.1))

    # Draw the shadow using the blurred alpha mask and shadow color
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.bitmap(shadow_offset, alpha_mask_blur, fill=shadow_color)

    # Adjust the opacity of the shadow image
    shadow = shadow.point(lambda p: p + shadow_opacity)
    image=image.convert('RGBA')
    # Composite the original image with the shadow using the alpha mask
    result = Image.alpha_composite(image, shadow)

    return result





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
    



