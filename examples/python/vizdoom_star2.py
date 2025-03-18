import vizdoom as vzd
import numpy as np
import heapq
import time

# ✅ Load ViZDoom Environment
def initialize_game():
    game = vzd.DoomGame()
    game.load_config("/ViZDoom/scenarios/basic.cfg")  # Ensure the correct path
    game.init()
    return game

# ✅ Define A* Algorithm
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(position, binary_map):
    x, y = position
    neighbors = [
        (x-1, y), (x+1, y), (x, y-1), (x, y+1),
        (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)
    ]
    return [(nx, ny) for nx, ny in neighbors if 0 <= nx < binary_map.shape[1] and 0 <= ny < binary_map.shape[0] and binary_map[ny, nx] == 1]

def astar(binary_map, start, goal):
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
            return path[::-1]

        for neighbor in get_neighbors(current_node.position, binary_map):
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

# ✅ Define Movement Actions
MOVE_FORWARD = [1, 0, 0, 0, 0, 0, 0]  # Move forward
TURN_LEFT = [0, 1, 0, 0, 0, 0, 0]  # Turn left
TURN_RIGHT = [0, 0, 1, 0, 0, 0, 0]  # Turn right

# ✅ Move Agent Along Path
def move_agent(game, path):
    for i in range(len(path) - 1):
        start_x, start_y = path[i]
        next_x, next_y = path[i + 1]

        if next_x > start_x:
            print("Turning right...")
            game.make_action(TURN_RIGHT, 5)  # Turn Right for 5 frames
        elif next_x < start_x:
            print("Turning left...")
            game.make_action(TURN_LEFT, 5)  # Turn Left for 5 frames

        print("Moving forward...")
        game.make_action(MOVE_FORWARD, 10)  # Move forward for 10 frames
        time.sleep(0.1)

# ✅ Main Execution
def main():
    game = initialize_game()
    
    # Load Binary Map
    binary_map = np.load("binary_map.npy")

    # Define Start and Goal Points
    start = (317, 132)
    goal = (345, 228)

    # Run A* Algorithm
    print("Finding path using A*...")
    path = astar(binary_map, start, goal)

    if path:
        print(f"Path found: {len(path)} steps")
        move_agent(game, path)
    else:
        print("No valid path found!")

    game.close()

if __name__ == "__main__":
    main()
