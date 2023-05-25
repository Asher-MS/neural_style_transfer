from noisy_edge import add_scale_noise
from PIL import Image
import sys

img=Image.open(sys.argv[1]).convert("RGB")

img=add_scale_noise(img,scale_factor=30,noise_intensity=1)


img.save("adipoli_temp.png")
