import vizdoom as vzd
import numpy as np
import heapq
import time
import math
import matplotlib.pyplot as plt

# ✅ Initialize ViZDoom with Window Display
def initialize_game():
    game = vzd.DoomGame()
    game.load_config("/ViZDoom/scenarios/basic.cfg")  # Ensure correct path

    # ✅ Ensure game window opens
    game.set_window_visible(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)  # High resolution
    game.set_mode(vzd.Mode.PLAYER)  # Enable player mode for interaction

    game.init()
    return game

# ✅ A* Pathfinding Algorithm
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

# ✅ Define Movement Actions Based on `basic.cfg`
ACTIONS = {
    "MOVE_FORWARD": [0] * 21,  # Ensure list matches `available_buttons`
    "TURN_LEFT": [0] * 21,
    "TURN_RIGHT": [0] * 21,
}

# ✅ Correct Button Indexes (based on basic.cfg)
ACTIONS["MOVE_FORWARD"][3] = 1  # MOVE_FORWARD is 4th in the list
ACTIONS["TURN_LEFT"][7] = 1      # TURN_LEFT is 8th in the list
ACTIONS["TURN_RIGHT"][8] = 1     # TURN_RIGHT is 9th in the list

# ✅ Function to Calculate Turn Direction
def get_turn_direction(agent_x, agent_y, target_x, target_y):
    """Returns whether the agent should turn left or right to face the next waypoint."""
    dx = target_x - agent_x
    dy = target_y - agent_y
    angle = math.atan2(dy, dx) * (180 / math.pi)  # Convert to degrees
    return "TURN_RIGHT" if angle > 0 else "TURN_LEFT"

# ✅ Move Agent Along A* Path
def move_agent(game, path):
    print(f"Following path with {len(path)} steps...")

    for i in range(len(path) - 1):
        current_x, current_y = path[i]
        next_x, next_y = path[i + 1]

        # Determine correct turning direction
        turn_direction = get_turn_direction(current_x, current_y, next_x, next_y)
        print(f"Turning {turn_direction.lower()}...")
        game.make_action(ACTIONS[turn_direction], 5)

        # Move forward after turning
        print("Moving forward...")
        game.make_action(ACTIONS["MOVE_FORWARD"], 10)
        time.sleep(0.1)

        # Debug state: Check if agent position updates
        state = game.get_state()
        if state:
            print(f"Agent Position Updated: {state.game_variables}")

# ✅ Visualize A* Path on Map
def visualize_path(binary_map, path, start, goal):
    plt.imshow(binary_map, cmap="gray", origin="upper")
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, color="red", linewidth=2, label="A* Path")
    plt.scatter([start[1]], [start[0]], color="yellow", label="Start")
    plt.scatter([goal[1]], [goal[0]], color="blue", label="Goal")
    plt.legend()
    plt.title("A* Path on Map")
    plt.show()

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
        
        # ✅ Show the computed path on a separate map window
        visualize_path(binary_map, path, start, goal)

        # ✅ Move the agent along the A* path
        move_agent(game, path)
    else:
        print("No valid path found!")


    game.close()

if __name__ == "__main__":
    main()
