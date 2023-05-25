import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from semantic_segmentation import paste_foreground_on_background
import os

import matplotlib.pyplot as plt
import time
import numpy as np
from PIL import Image
import sys
from style_transfer import transfer_style
from semantic_segmentation import separate_foreground_background

content_image=sys.argv[1]
style_image=sys.argv[2]

content_fr,content_bg=separate_foreground_background(content_image)
style_fr,style_bg=separate_foreground_background(style_image)

content_fr.save("temp/content_fr.jpg")
content_bg.save("temp/content_bg.jpg")
style_fr.save("temp/style_fr.jpg")
style_bg.save("temp/style_bg.jpg")


transfer_style(content_fr,style_fr,save_image=False,change_background=False).save("foreground_output.jpg")
transfer_style(content_bg,style_bg,save_image=False,change_background=False).save("background_output.jpg")


