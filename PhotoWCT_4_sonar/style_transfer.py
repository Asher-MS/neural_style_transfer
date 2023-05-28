import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from semantic_segmentation import paste_foreground_on_background
import os

from models import Sonar_noise_WCT
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image
import sys



def transfer_style(content_image,style_image,save_image=True,change_background=True,background_image=None,output_name=None,rotate_angle=0,add_noise=False):
    # Load model
    model_name = './PhotoWCTModels/photo_wct.pth'

    sn_wct = Sonar_noise_WCT()
    sn_wct.load_state_dict(torch.load(model_name))

    sn_wct.cuda()


    if change_background:
        cont_img,alpha_mask = paste_foreground_on_background(content_image,background_image,rotate_angle=rotate_angle,add_noise=add_noise)
        cont_img=cont_img.convert('RGB')
    else:
        if isinstance(content_image,Image.Image):
            cont_img=content_image
        else:
            cont_img=Image.open(content_image).convert('RGB')
    cont_img = cont_img.resize((256, 256))
    cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
    cont_img = cont_img.cuda()


    background_image=Image.open(background_image)

    if isinstance(style_image,Image.Image):
        styl_img=style_image
    else:
        styl_img=Image.open(style_image).convert('RGB')
    styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)
    styl_img = styl_img.cuda()

    with torch.no_grad():  
        stylized_img = sn_wct.transform(cont_img, styl_img, 1.0)
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        alpha_mask=alpha_mask.convert("RGBA")
        out_img=out_img.resize(background_image.size,Image.ANTIALIAS)
        background_image.paste(out_img,(0,0),mask=alpha_mask)
        out_img=background_image
        #### save img
    
        if save_image:
            out_img.save(output_name)

    return out_img
        








if __name__== "__main__":
    help_message="""Example:python style_transfer.py <content image path> <style image path> <background image> <output path>"""

    try:
        content_image_path = sys.argv[1]
        style_image_path = sys.argv[2]
        output_image_path=sys.argv[4]
        try:
            background_image=sys.argv[3]
        except Exception as e:
            print("Sea Floor not provided,using default sea floor")
            background_image="./data/seafloor/floor-1.png"

    except Exception as e:
        print("Not enough arguments")
        print(help_message)
        exit()
    transfer_style(content_image_path,style_image_path,background_image=background_image,output_name=output_image_path)
