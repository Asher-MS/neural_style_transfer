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


output_image_path = './data/output/'
def transfer_style(content_image_path,style_image_path,background_image,output_name):
    # Load model
    model_name = './PhotoWCTModels/photo_wct.pth'

    sn_wct = Sonar_noise_WCT()
    sn_wct.load_state_dict(torch.load(model_name))

    sn_wct.cuda()


    cont_img = paste_foreground_on_background(content_image_path,background_image).convert('RGB')
    cont_img = cont_img.resize((256, 256))
    cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
    cont_img = cont_img.cuda()




    styl_img = Image.open(style_image_path).convert('RGB')
    styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)
    styl_img = styl_img.cuda()

    with torch.no_grad():  
        stylized_img = sn_wct.transform(cont_img, styl_img, 1.0)
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        #### save img

        out_img.save(output_name)
        


if __name__== "__main__":
    help_message="""Example:python style_transfer.py <content image path> <style image path>"""

    try:
        content_image_path = sys.argv[1]
        style_image_path = sys.argv[2]
        try:
            background_image=sys.argv[3]
        except Exception as e:
            print("Sea Floor not provided,using default sea floor")
            background_image="./data/seafloor/floor-1.png"

    except Exception as e:
        print("Not enough arguments")
        print(help_message)
        exit()



    
    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)
        transfer_style(content_image_path,style_image_path,background_image)
