import cv2
import numpy as np

# Load the map image
image_path = "/ViZDoom/examples/python/map_full.png"  # Adjust the path if needed
map_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if map_image is None:
    raise ValueError("Error: Could not load the map image. Check the file path.")

# Apply Otsu's Thresholding to binarize the image
_, binary_map = cv2.threshold(map_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save the binary map for debugging
binary_map_path = "/ViZDoom/examples/python/binary_map.png"
cv2.imwrite(binary_map_path, binary_map)

print(f"Binary map saved to: {binary_map_path}")
