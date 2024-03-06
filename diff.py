import cv2
import numpy as np

# Load images
image1 = cv2.imread("images/frame3.png")
image2 = cv2.imread("images/frame2.png")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute difference
difference = cv2.absdiff(image1, image2)

# Convert to grayscale and threshold to get a binary mask of differences
_, diff_mask = cv2.threshold(difference, 5, 255, cv2.THRESH_BINARY_INV)

# Create an ellipse mask
ellipse_mask = np.zeros_like(difference)
center = (ellipse_mask.shape[1] // 2,
          ellipse_mask.shape[0] // 2)  # Image center
axes_length = (300, 150)  # Customize the size of your ellipse here
cv2.ellipse(ellipse_mask, center, axes_length, 0, 0, 360, (255, 255, 255), -1)

# Combine the difference mask and the ellipse mask to get the final mask
final_mask = cv2.bitwise_and(diff_mask, ellipse_mask)

# Create an image with a black background
black_background = np.zeros_like(image1)

# Apply the final mask to make the changes appear white within the ellipse
# Note: Since we want white, we set all channels to 255 where there are changes
black_background[final_mask != 0] = 255

# The resulting image (black_background) now has a black background with white changes
cv2.imwrite('images/diff_Ellipse_White_Changes.png', black_background)
cv2.imwrite('images/difference.png', difference)
cv2.imwrite('images/diff_mask.png', diff_mask)
