import vizdoom as vzd
import numpy as np
import heapq
import math
import time
import matplotlib.pyplot as plt

# ✅ Load Binary Map
binary_map = np.load("binary_map.npy")

# ✅ Define Start and Goal in A* Map
start = (317, 132)  # Spawn point in binary map coordinates
goal = (345, 228)   # Goal point in binary map coordinates

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
    """Returns valid neighbors ensuring diagonal moves do not pass through walls."""
    x, y = position
    height, width = binary_map.shape
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonals
    ]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and binary_map[ny, nx] == 1:
            if dx != 0 and dy != 0:  # Diagonal movement check
                if binary_map[y, nx] == 0 or binary_map[ny, x] == 0:
                    continue
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
    return None

# ✅ Compute A* Path
path = astar(start, goal)

# ✅ Convert A* Path to ViZDoom Coordinates
def convert_coordinates(x, y):
    scale_x = 18.25
    scale_y = -18.0104
    x_offset = 317
    y_offset = 132
    offset_y = -64
    vizdoom_x = (x - x_offset) * scale_x
    vizdoom_y = - (y - y_offset) * scale_y + offset_y
    return vizdoom_x, vizdoom_y

vizdoom_path = [convert_coordinates(x, y) for x, y in path] if path else []

# ✅ Initialize ViZDoom
def initialize_game():
    game = vzd.DoomGame()
    game.load_config("basic.cfg")  # Ensure basic.cfg is in the working directory
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game

# ✅ Movement Commands
ACTIONS = {
    "MOVE_FORWARD": [0] * 20,
    "TURN_LEFT": [0] * 20,
    "TURN_RIGHT": [0] * 20
}
ACTIONS["MOVE_FORWARD"][6] = 1   # MOVE_FORWARD (index based on basic.cfg buttons)
ACTIONS["TURN_LEFT"][8] = 1     # TURN_LEFT
ACTIONS["TURN_RIGHT"][7] = 1    # TURN_RIGHT

# ✅ Move Agent Along Path
def move_along_path(game, path):
    print(f"🚀 Moving agent along path with {len(path)} steps...")
    for i in range(len(path) - 1):
        state = game.get_state()
        if state is None:
            print("⚠️ No state received! Exiting...")
            return

        game_vars = state.game_variables
        if len(game_vars) < 3:
            print("⚠️ Warning: Expected 3 game variables (X, Y, ANGLE), but got", len(game_vars))
            return

        agent_x, agent_y, agent_angle = game_vars[:3]
        next_x, next_y = path[i + 1]

        # Compute Desired Angle
        dx = next_x - agent_x
        dy = next_y - agent_y
        target_angle = math.degrees(math.atan2(dy, dx))

        # Normalize angles to [0, 360)
        target_angle = target_angle % 360
        agent_angle = agent_angle % 360

        # Rotate Agent (shortest direction)
        angle_diff = (target_angle - agent_angle + 180) % 360 - 180
        while abs(angle_diff) > 5:
            action = ACTIONS["TURN_LEFT"] if angle_diff > 0 else ACTIONS["TURN_RIGHT"]
            game.make_action(action, 1)  # 1 tic per action
            state = game.get_state()
            agent_angle = state.game_variables[2] % 360
            angle_diff = (target_angle - agent_angle + 180) % 360 - 180

        # Move Forward
        distance = math.sqrt(dx * dx + dy * dy)
        steps = int(distance / 10) if distance > 0 else 1  # Assume 10 units per tic (adjust if needed)
        for _ in range(min(steps, 100)):  # Limit to 100 steps to avoid infinite loops
            game.make_action(ACTIONS["MOVE_FORWARD"], 1)
            state = game.get_state()
            current_x, current_y = state.game_variables[:2]
            if math.sqrt((current_x - next_x) ** 2 + (current_y - next_y) ** 2) < 10:
                break
        print(f"✅ Moved to: ({next_x:.1f}, {next_y:.1f}) | Current: ({current_x:.1f}, {current_y:.1f})")

# ✅ Main Execution
def main():
    game = initialize_game()

    if path:
        print(f"✅ Path found: {len(path)} steps")

        # ✅ Debugging Path Min-Max Values
        min_x, max_x = min(x for x, _ in path), max(x for x, _ in path)
        min_y, max_y = min(y for _, y in path), max(y for _, y in path)
        print(f"A* Path Min-Max X: {min_x} to {max_x}")
        print(f"A* Path Min-Max Y: {min_y} to {max_y}")

        # ✅ Show Path on Binary Map
        plt.figure(figsize=(8, 6))
        plt.imshow(binary_map, cmap="gray", origin="upper")
        plt.scatter(*zip(*path), color="red", s=3, label="A* Path")
        plt.scatter(start[0], start[1], color="yellow", label="Start")
        plt.scatter(goal[0], goal[1], color="blue", label="Goal")
        plt.legend()
        plt.title("A* Path on Binary Map")
        plt.show()

        # ✅ Move Agent
        move_along_path(game, vizdoom_path)
    else:
        print("❌ No path found!")

    game.close()

if __name__ == "__main__":
    main()