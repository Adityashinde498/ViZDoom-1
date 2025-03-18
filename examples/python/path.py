import numpy as np
import heapq
import matplotlib.pyplot as plt

# Load binary map
binary_map = np.load("binary_map.npy")

# Define movement directions (4-way: Up, Down, Left, Right)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Can add diagonals for 8-way

def heuristic(a, b):
    """Compute Euclidean distance heuristic."""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(start, goal, binary_map):
    rows, cols = binary_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))  # (f-cost, (x, y))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for move in MOVES:
            neighbor = (current[0] + move[0], current[1] + move[1])

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and binary_map[neighbor] == 0:
                tentative_g_score = g_score[current] + 1  # Assuming uniform cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

# Run A* algorithm
path = a_star((317.308, 132.922), (349.648, 229.427), binary_map)

# Visualize the result
plt.imshow(binary_map, cmap="gray")
if path:
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, marker="o", color="red", markersize=3, linestyle="-")  # Path visualization
plt.scatter([317.308], [132.922], color="yellow", marker="s", label="Start")
plt.scatter([349.648], [229.427], color="blue", marker="s", label="Goal")
plt.legend()
plt.title("A* Pathfinding")
plt.show()
