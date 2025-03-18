import cv2
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# Load the map image in color
map_color = cv2.imread("/ViZDoom/examples/python/map_full.png")  # Adjust path as needed

# Find the white diamond (start) - assuming white is [255, 255, 255] in BGR
white_pixels = np.where(np.all(map_color == [255, 255, 255], axis=-1))
start_y, start_x = white_pixels[0][0], white_pixels[1][0]
start = (start_x, start_y)

# Find the blue square (target) - assuming blue is [255, 0, 0] in BGR
blue_pixels = np.where(np.all(map_color == [255, 0, 0], axis=-1))
target_y, target_x = blue_pixels[0][0], blue_pixels[1][0]
target = (target_x, target_y)

# Load the map in grayscale for the occupancy grid
map_gray = cv2.imread("/ViZDoom/examples/python/map_full.png", cv2.IMREAD_GRAYSCALE)

# Create a binary occupancy grid (0 for free space, 1 for obstacles)
_, occupancy_grid = cv2.threshold(map_gray, 200, 1, cv2.THRESH_BINARY_INV)

# Invert the grid for pathfinding (1 for walkable, 0 for obstacles)
navigable_grid = 1 - occupancy_grid

# Create a pathfinding grid
grid = Grid(matrix=navigable_grid)

# Define start and target nodes
start_node = grid.node(start[0], start[1])  # (x, y) = (column, row)
end_node = grid.node(target[0], target[1])

# Find the path using A*
finder = AStarFinder()
path, _ = finder.find_path(start_node, end_node, grid)

# Convert path to list of (x, y) coordinates
path_coords = [(node.x, node.y) for node in path]

# Print results
print(f"Start (white diamond): {start}")
print(f"Target (blue square): {target}")
print("Occupancy grid shape:", occupancy_grid.shape)
print("Path length:", len(path_coords))
print("Path coordinates:", path_coords)