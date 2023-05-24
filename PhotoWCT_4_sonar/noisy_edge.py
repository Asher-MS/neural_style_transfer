
import cv2
import numpy as np


def add_scale_noise(image, scale_factor, noise_intensity):

    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    resized_edges = cv2.resize(
        dilated_edges, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    _, mask = cv2.threshold(resized_edges, 0, 255, cv2.THRESH_BINARY)

    noisy_mask = cv2.GaussianBlur(mask.astype(
        np.float32), (0, 0), noise_intensity)

    noisy_mask = cv2.resize(noisy_mask, (image.shape[1], image.shape[0]))

    noisy_image = np.copy(image)
    for channel in range(3):
        noisy_image[:, :, channel] = (
            image[:, :, channel] * (1 - noisy_mask)).astype(np.uint8)

    # cv2.imwrite(content_image_path+'_noise.jpg', noisy_image)
    return noisy_image


# # Load the image
# image = cv2.imread('./output.jpg')

# if image is None:
#     print(f"Failed to load image at path: {image_path}")
# else:
#     # Add scale noise to the edges
#     scale_factor = 30  # Adjust the scale factor as desired
#     noise_intensity = 1  # Adjust the noise intensity as desired
#     noisy_image = add_scale_noise(image, scale_factor, noise_intensity)

# # Display the noisy image
# output_path = './data/output/out.jpg'
# cv2.imwrite(output_path, noisy_image)
# print(f"Noisy image saved at path: {output_path}")
