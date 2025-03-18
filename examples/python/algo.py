import numpy as np
import matplotlib.pyplot as plt
import heapq

# Load binary map
binary_map = np.load("binary_map.npy")

# Define start and goal points
start = (317, 132)
goal = (345, 228)

# A* Algorithm Implementation
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic (estimated cost from this node to goal)
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))  # Euclidean distance

def get_neighbors(position, binary_map):
    """
    Returns valid neighbors ensuring diagonal moves do not pass through walls.
    """
    x, y = position
    height, width = binary_map.shape
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
        (-1, -1), (-1, 1), (1, -1), (1, 1) # Diagonals
    ]
    
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        
        # Ensure within map bounds
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        
        # Ensure destination is walkable
        if binary_map[ny, nx] != 1:
            continue
        
        # Block diagonal moves that would pass through walls
        if dx != 0 and dy != 0:  # Diagonal movement check
            if binary_map[y, nx] == 0 or binary_map[ny, x] == 0:
                continue  # Diagonal move is blocked
        
        neighbors.append((nx, ny))
    
    return neighbors

def astar(start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        for neighbor in get_neighbors(current_node.position, binary_map):  # Use updated get_neighbors
            if neighbor in closed_set:
                continue

            neighbor_node = Node(neighbor, current_node)
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic(neighbor, goal)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if any(open_node.position == neighbor and open_node.f <= neighbor_node.f for open_node in open_list):
                continue

            heapq.heappush(open_list, neighbor_node)

    return None  # No path found

# Find the path
path = astar(start, goal)

# Debugging: Visualize binary map
plt.figure(figsize=(8, 6))
plt.imshow(binary_map, cmap="gray", origin="upper")
plt.scatter(start[0], start[1], color="yellow", label="Start")
plt.scatter(goal[0], goal[1], color="blue", label="Goal")
if path:
    plt.plot(*zip(*path), color="red", linewidth=2, label="A* Path")
else:
    print("No path found")
plt.legend()
plt.title("A* Path on Binary Map (Fixed)")
plt.show()
