import numpy as np
from PIL import Image

# Load the image
img = Image.open('image.jpg')
# img_arr = np.asarray(img)


img_arr=np.array([[1,2,3],[4,5,6],[7,8,9]])

# Convert the image to grayscale
# img_gray = np.mean(img_arr, axis=2)

img_gray=img_arr
# Subtract the mean pixel value
img_gray = img_gray- np.mean(img_gray)
print("Centered Matrix")
print(img_gray)

# Compute the covariance matrix
cov_mat = np.cov(img_gray, rowvar=False)
print("Covariance matrix")
print(cov_mat)


# Compute the eigendecomposition of the covariance matrix
eigvals, eigvecs = np.linalg.eigh(cov_mat)

# Diagonalize the covariance matrix using the inverse square root of the eigenvalues
whiten_mat = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-5)) @ eigvecs.T
print("Whiten Matrix")
print(whiten_mat)

# Apply the whitening matrix to the image
img_white = img_gray @ whiten_mat

print("Whitened MAtrix")
print(img_white)


# Convert the whitened image back to uint8 format
# img_white = np.uint8((img_white - np.min(img_white)) / (np.max(img_white) - np.min(img_white)) * 255)

# Save the whitened image
# Image.fromarray(img_white).save('image_white.jpg')