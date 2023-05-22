import cv2
import numpy as np
import sys
# Load the texture image
texture = cv2.imread(sys.argv[2])

# Load the target image to apply the texture
target_image = cv2.imread(sys.argv[1])

# Resize the texture to match the size of the target image
texture = cv2.resize(texture, (target_image.shape[1], target_image.shape[0]))

# Convert the texture to grayscale
# texture_gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)

# Apply the texture to the target image using a blending function
blended_image = cv2.addWeighted(target_image, 0.9, texture, 0.8, 0)

# Display the blended image
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
