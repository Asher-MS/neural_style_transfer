import os
from style_transfer import transfer_style
import sys


content_dir="./data/plane_op/"
style_image="./data/plane/plane-001.png"
output_dir="./data/output/planes/"
background_image="./data/seafloor/floor-31.png"

temp=0
for i in os.listdir(content_dir):
    transfer_style(content_image=content_dir+i,style_image=style_image,background_image=background_image,output_name=output_dir+str(temp)+".jpg",rotate_angle=0,add_noise=True)
    transfer_style(content_image=content_dir+i,style_image=style_image,background_image=background_image,output_name=output_dir+str(temp+1)+".jpg",rotate_angle=90,add_noise=True)
    transfer_style(content_image=content_dir+i,style_image=style_image,background_image=background_image,output_name=output_dir+str(temp+2)+".jpg",rotate_angle=180,add_noise=True)
    transfer_style(content_image=content_dir+i,style_image=style_image,background_image=background_image,output_name=output_dir+str(temp+3)+".jpg",rotate_angle=270,add_noise=True)
    temp=temp+4
