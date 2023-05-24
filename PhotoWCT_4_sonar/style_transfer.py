import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from semantic_segmentation import paste_foreground_on_background
from noisy_edge import add_scale_noise
import os

from models import Sonar_noise_WCT
import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image
import sys


output_image_path = './data/output/'


def transfer_style(content_image_path, style_image_path, background_image, output_name):
    # Load model
    model_name = './PhotoWCTModels/photo_wct.pth'

    sn_wct = Sonar_noise_WCT()
    sn_wct.load_state_dict(torch.load(model_name))

    sn_wct.cuda()

    cont_imgs = paste_foreground_on_background(
        content_image_path, background_image, output_name)

    cont_img = cont_imgs[0].convert('RGB')
    cont_img = cont_img.resize((256, 256))
    cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
    cont_img = cont_img.cuda()

    scale_factor = 30
    noise_intensity = 1
    noisy_mask = add_scale_noise(cont_imgs[1].convert(
        'RGB'), scale_factor, noise_intensity)
    noisy_mask = Image.fromarray(noisy_mask)

    masked_cont = paste_foreground_on_background(
        noisy_mask, background_image, output_name)
    # cont_img_n = np.array(masked_cont[0].convert('RGB'))
    # if not '_noise' in content_image_path:
    #     cp = content_image_path.split('.j')

    # cont_img_noise = Image.fromarray(cont_img_n)
    cont_img_noise = masked_cont[0].convert('RGB')
    cont_img_noise = cont_img_noise.resize((256, 256))
    cont_img_noise = transforms.ToTensor()(cont_img_noise).unsqueeze(0)
    cont_img_noise = cont_img_noise.cuda()

    styl_img = Image.open(style_image_path).convert('RGB')
    styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)
    styl_img = styl_img.cuda()

    with torch.no_grad():
        stylized_img = sn_wct.transform(cont_img, styl_img, 1.0)
        stylized_img_noise = sn_wct.transform(cont_img_noise, styl_img, 1.0)
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        gridn = utils.make_grid(stylized_img_noise.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(
            0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)
        ndarr_n = gridn.mul(255).clamp(
            0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img_n = Image.fromarray(ndarr_n)

        # save img

        out_img.save(output_name)
        o = output_name.split('.j')
        out_img_n.save(o[0]+'_noise.jpg')


if __name__ == "__main__":
    help_message = """Example:python style_transfer.py <content image path> <style image path>"""

    try:
        content_image_path = sys.argv[1]
        style_image_path = sys.argv[2]
        try:
            background_image = sys.argv[3]
        except Exception as e:
            print("Sea Floor not provided,using default sea floor")
            background_image = "./data/seafloor/floor-1.png"

    except Exception as e:
        print("Not enough arguments")
        print(help_message)
        exit()

    if not os.path.exists(output_image_path):
        os.mkdir(output_image_path)
        transfer_style(content_image_path, style_image_path, background_image)
