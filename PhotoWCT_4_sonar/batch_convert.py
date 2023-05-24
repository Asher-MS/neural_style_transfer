import os
from style_transfer import transfer_style
import sys


content_dir = "./data/content/"
style_image = "./data/plane/plane-010.png"
output_dir = "./data/output/"
background_image = "./data/seafloor/floor-10.png"

temp = 0
for i in os.listdir(content_dir):
    transfer_style(content_dir+i, style_image, background_image,
                   output_dir+str(temp)+".jpg")
    temp = temp+1
