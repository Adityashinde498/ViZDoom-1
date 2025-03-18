import cv2
import numpy as np

# Load the image
image_path = "map_full.png"
image = cv2.imread(image_path)

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges
white_lower = np.array([0, 0, 200], dtype=np.uint8)
white_upper = np.array([180, 30, 255], dtype=np.uint8)

blue_lower = np.array([100, 150, 50], dtype=np.uint8)
blue_upper = np.array([140, 255, 255], dtype=np.uint8)

# Threshold the image
white_mask = cv2.inRange(hsv, white_lower, white_upper)
blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

# Get coordinates of white pixels
white_coords = np.column_stack(np.where(white_mask > 0))
blue_coords = np.column_stack(np.where(blue_mask > 0))

# Select first detected coordinates
if len(white_coords) > 0:
    start_point = tuple(white_coords[0][::-1])  # Reverse (row, col) to (x, y)
else:
    start_point = None

if len(blue_coords) > 0:
    goal_point = tuple(blue_coords[0][::-1])  # Reverse (row, col) to (x, y)
else:
    goal_point = None

print("Start Point (White):", start_point)
print("Goal Point (Blue):", goal_point)
