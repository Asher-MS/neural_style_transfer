import os
from style_transfer import transfer_style
import sys


content_dir="./data/ship_op/"
style_image="./data/plane/plane-001.png"
output_dir="./data/output/ships/"
background_image="./data/seafloor/floor-18.png"

temp=0
for i in os.listdir(content_dir):
    transfer_style(content_image=content_dir+i,style_image=style_image,background_image=background_image,output_name=output_dir+str(temp)+".jpg")
    temp=temp+1
