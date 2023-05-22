from PIL import Image
import requests
from rembg import remove
import sys
import cv2
import numpy as np


# def add_scale_noise(image, scale_factor, noise_intensity):

#     image = np.array(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     # Check if the image is grayscale
#     if len(image.shape) == 2:
#         # Add an extra dimension for compatibility with color images
#         image = np.expand_dims(image, axis=2)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Canny edge detection to obtain the edges
#     edges = cv2.Canny(gray, 50, 150)

#     # Dilate the edges to make them thicker
#     kernel = np.ones((3, 3), np.uint8)
#     dilated_edges = cv2.dilate(edges, kernel, iterations=1)

#     # Resize the dilated edges to introduce scale noise
#     resized_edges = cv2.resize(
#         dilated_edges, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

#     # Threshold the resized edges to create a binary mask
#     _, mask = cv2.threshold(resized_edges, 0, 255, cv2.THRESH_BINARY)

#     # Add Gaussian noise to the edges
#     noisy_mask = cv2.GaussianBlur(mask.astype(
#         np.float32), (0, 0), noise_intensity)

#     # Rescale the noisy mask to match the original image size
#     noisy_mask = cv2.resize(noisy_mask, (image.shape[1], image.shape[0]))

#     # Apply the noisy mask to the original image
#     noisy_image = np.copy(image)
#     for channel in range(3):
#         noisy_image[:, :, channel] = (
#             image[:, :, channel] * (1 - noisy_mask)).astype(np.uint8)
#     noisy_image = Image.fromarray(noisy_image, 'RGB')
#     return noisy_image


def paste_foreground_on_background(foreground_image, background_path, output_path):

    foreground_image = Image.open(foreground_image)

    if foreground_image.mode != "RGBA":
        foreground_image = foreground_image.convert("RGBA")

    with remove(foreground_image) as alpha_mask:

        background_image = Image.open(background_path)

        foreground_image = foreground_image.resize(
            background_image.size, Image.ANTIALIAS)

        if foreground_image.mode != background_image.mode:
            foreground_image = foreground_image.convert(background_image.mode)

        alpha_mask = alpha_mask.resize(background_image.size, Image.ANTIALIAS)

        # # alpha_mask = alpha_mask.convert("L")
        # # alpha_mask.save('mask.jpg')
        # # mask_img = Image.open('./mask.jpg')
        # mask_img = Image.open('./o6.jpg')
        background_image.paste(mask_img, (0, 0), mask=alpha_mask)
        background_image.save(output_path)


def add_scale_noise(image, scale_factor, noise_intensity):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to obtain the edges
    edges = cv2.Canny(gray, 50, 150)

    # Dilate the edges to make them thicker
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Resize the dilated edges to introduce scale noise
    resized_edges = cv2.resize(
        dilated_edges, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    # Threshold the resized edges to create a binary mask
    _, mask = cv2.threshold(resized_edges, 0, 255, cv2.THRESH_BINARY)

    # Add Gaussian noise to the edges
    noisy_mask = cv2.GaussianBlur(mask.astype(
        np.float32), (0, 0), noise_intensity)

    # Rescale the noisy mask to match the original image size
    noisy_mask = cv2.resize(noisy_mask, (image.shape[1], image.shape[0]))

    # Apply the noisy mask to the original image
    noisy_image = np.copy(image)
    for channel in range(3):
        noisy_image[:, :, channel] = (
            image[:, :, channel] * (1 - noisy_mask)).astype(np.uint8)

    return noisy_image


foreground_image = sys.argv[1]
background_path = sys.argv[2]
output_path = "output.jpg"

paste_foreground_on_background(foreground_image, background_path, output_path)
# Load the image
image = cv2.imread('mask.jpg')

if image is None:
    print(f"Failed to load image at path: {output_path}")
else:
    scale_factor = 10
    noise_intensity = 1
    noisy_image = add_scale_noise(image, scale_factor, noise_intensity)


noise_output_path = 'noise_out.jpg'
cv2.imwrite(noise_output_path, noisy_image)
print(f"Noisy image saved at path: {noise_output_path}")
