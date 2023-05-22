# import matplotlib.pyplot as plt
# import torch
# from torch.nn.functional import conv2d
# import numpy as np
# from scipy.signal import gaussian
# import sys
# from PIL import Image


# def add_scaly_noise(image, scale=0.1, sigma=5):
#     # Generate a scaly noise kernel
#     kernel_size = int(scale * min(image.shape[1], image.shape[2]))
#     kernel = gaussian(kernel_size, sigma=sigma).reshape(
#         1, 1, kernel_size, kernel_size)
#     kernel = torch.from_numpy(kernel).float()

#     # Pad the image to ensure that the convolution output has the same size as the input
#     padding = kernel_size // 2
#     image = torch.nn.functional.pad(
#         image, (padding, padding, padding, padding), mode='reflect')

#     # Apply the convolution operation to add scaly noise to the edges of the image
#     noise = conv2d(image, kernel, stride=1, padding=0)
#     return image + noise


# # Load an example image
# # image = torch.rand(1, 3, 256, 256)
# img = sys.argv[1]
# image = Image.open(img)
# # Add scaly noise to the edges of the image
# # image = torch.from_numpy(np.array(image)).permute(
# #     2, 0, 1).float().unsqueeze(0) / 255.0

# image = torch.from_numpy(np.array(image)).float().permute(
#     2, 0, 1).unsqueeze(0) / 255.0
# noisy_image = add_scaly_noise(image)

# # Visualize the original and noisy images
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(image[0].permute(1, 2, 0))
# ax[1].imshow(noisy_image[0].permute(1, 2, 0))
# plt.show()

import cv2
import numpy as np


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


# Load the image
image = cv2.imread('./output.jpg')

if image is None:
    print(f"Failed to load image at path: {image_path}")
else:
    # Add scale noise to the edges
    scale_factor = 30  # Adjust the scale factor as desired
    noise_intensity = 1  # Adjust the noise intensity as desired
    noisy_image = add_scale_noise(image, scale_factor, noise_intensity)

# Display the noisy image
output_path = './data/output/out.jpg'
cv2.imwrite(output_path, noisy_image)
print(f"Noisy image saved at path: {output_path}")
