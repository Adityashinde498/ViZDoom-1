import cv2
import numpy as np

# Load the binary image
image = cv2.imread("map_full.png", cv2.IMREAD_GRAYSCALE)

# Apply thresholding (just in case the image isn't pure binary)
_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Remove small artifacts (noise) using morphological opening
kernel = np.ones((3,3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# Save the cleaned binary map
cv2.imwrite("binary_map_cleaned.png", cleaned)

print("Cleaned binary map saved as binary_map_cleaned.png")


