import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary map image
image_path = "/ViZDoom/examples/python/fixed_binary_map.png"  # Update this path if needed
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Convert to binary: Walls (black) = 0, Free space (white) = 1
#_, binary_map = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

#binary_map = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                   cv2.THRESH_BINARY, 11, 2)
image = cv2.resize(image, (450, 300))



_, binary_map = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
#_, binary_map = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

#kernel = np.ones((3,3), np.uint8)  # Adjust size if needed
#binary_map = cv2.dilate(binary_map, kernel, iterations=1)



# Convert to NumPy array with 0 (walls) and 1 (free paths)
binary_array = (binary_map == 255).astype(np.uint8)

# Display the converted map
plt.imshow(binary_array, cmap='gray')
plt.title("Binary Map")
plt.show()

# Save the array for future use
np.save("binary_map.npy", binary_array)

print("Binary map conversion completed! Saved as 'binary_map.npy'.")
