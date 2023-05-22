import cv2
import numpy as np
import sys
# Load the SSS image of the plane
image = cv2.imread(sys.argv[1])

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a texture extraction method (e.g., Gabor filters, Local Binary Patterns, etc.)
# Here, we'll use the OpenCV implementation of Gabor filters
ksize = (31, 31)  # Kernel size of the Gabor filter
sigma = 5.0       # Standard deviation of the Gaussian envelope
theta = 0         # Orientation of the Gabor filter
lambd = 10        # Wavelength of the sinusoidal factor
gamma = 0.5       # Spatial aspect ratio
psi = 0           # Phase offset
gabor_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi)

# Apply the Gabor filter to the grayscale image
filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)

# Threshold the filtered image to obtain the texture mask
_, texture_mask = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply the texture mask to the original image to extract the texture
texture = cv2.bitwise_and(image, image, mask=texture_mask)

# Display the texture image
# cv2.imshow('Texture Image', texture)
cv2.imwrite('texture.jpg',texture)
cv2.waitKey(0)
cv2.destroyAllWindows()
