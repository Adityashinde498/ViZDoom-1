import vizdoom as vzd
import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
import time
import random

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
            # More strict diagonal movement check
            if dx != 0 and dy != 0:  # Diagonal movement
                # Check both adjacent cells to ensure no corner cutting
                if binary_map[y, nx] == 0 or binary_map[ny, x] == 0:
                    continue
                # Buffer check for passable corners
                buffer_size = 2
                corner_passable = True
                for bx in range(max(0, nx - buffer_size), min(width, nx + buffer_size + 1)):
                    for by in range(max(0, ny - buffer_size), min(height, ny + buffer_size + 1)):
                        if binary_map[by, bx] == 0:
                            dist = math.sqrt((bx - nx) ** 2 + (by - ny) ** 2)
                            if dist <= buffer_size:
                                corner_passable = False
                                break
                    if not corner_passable:
                        break
                if not corner_passable:
                    continue
            neighbors.append((nx, ny))
    return neighbors

def astar(start, goal, binary_map):
    open_list = []
    closed_set = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)
    
    all_nodes = {start: start_node}

    while open_list:
        current_node = heapq.heappop(open_list)
        
        if current_node.position in closed_set:
            continue
            
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

            dx = neighbor[0] - current_node.position[0]
            dy = neighbor[1] - current_node.position[1]
            move_cost = 1.414 if dx != 0 and dy != 0 else 1
            tentative_g = current_node.g + move_cost
            
            if neighbor in all_nodes and tentative_g >= all_nodes[neighbor].g:
                continue
                
            if neighbor in all_nodes:
                neighbor_node = all_nodes[neighbor]
                neighbor_node.parent = current_node
                neighbor_node.g = tentative_g
                neighbor_node.f = tentative_g + neighbor_node.h
            else:
                neighbor_node = Node(neighbor, current_node)
                neighbor_node.g = tentative_g
                neighbor_node.h = heuristic(neighbor, goal)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                all_nodes[neighbor] = neighbor_node
                
            heapq.heappush(open_list, neighbor_node)
                
    return None

# ✅ Improved Coordinate Conversion with Better Calibration
def calibrate_coordinates():
    # Initial calibration based on known points
    spawn_x, spawn_y = 0.0, -64.0  # Expected ViZDoom spawn
    map_spawn = (317, 132)
    
    # Initial scale values (adjustable)
    scale_x = 18.25
    scale_y = -18.0  # Adjust based on actual range
    
    # Compute offsets
    doom_x_offset = spawn_x - (map_spawn[0] - 225) * scale_x / 1.1356  # Center adjustment
    doom_y_offset = spawn_y - (map_spawn[1] - 150) * scale_y / 5.7633  # Center adjustment
    
    print(f"🔍 Calibration Results:")
    print(f"  Scale X: {scale_x}, Scale Y: {scale_y}")
    print(f"  Map Offsets: {map_spawn}")
    print(f"  Doom Offsets: ({doom_x_offset:.1f}, {doom_y_offset:.1f})")
    
    return scale_x, scale_y, map_spawn[0], map_spawn[1], doom_x_offset, doom_y_offset

def convert_coordinates(x, y, calibration=None):
    if calibration is None:
        calibration = calibrate_coordinates()
    
    scale_x, scale_y, map_offset_x, map_offset_y, doom_x_offset, doom_y_offset = calibration
    
    vizdoom_x = doom_x_offset + (x - map_offset_x) * scale_x
    vizdoom_y = doom_y_offset + (y - map_offset_y) * scale_y
    
    return vizdoom_x, vizdoom_y

# ✅ Initialize ViZDoom with Enhanced Configuration
def initialize_game():
    game = vzd.DoomGame()
    game.load_config("basic.cfg")
    
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    
    game.clear_available_buttons()
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)
    
    game.add_game_args("+freelook 1")
    game.add_game_args("+cl_run 1")
    game.add_game_args("+sv_noautoaim 1")
    game.add_game_args("+sv_nocrouch 1")
    game.add_game_args("+sv_nojump 1")
    
    game.init()
    return game

# ✅ Movement Commands
ACTIONS = {
    "MOVE_FORWARD": [1, 0, 0, 0, 0, 0],
    "MOVE_BACKWARD": [0, 1, 0, 0, 0, 0],
    "TURN_RIGHT": [0, 0, 1, 0, 0, 0],
    "TURN_LEFT": [0, 0, 0, 1, 0, 0],
    "STRAFE_RIGHT": [0, 0, 0, 0, 1, 0],
    "STRAFE_LEFT": [0, 0, 0, 0, 0, 1],
    "FORWARD_RIGHT": [1, 0, 1, 0, 0, 0],
    "FORWARD_LEFT": [1, 0, 0, 1, 0, 0],
    "STRAFE_FORWARD_RIGHT": [1, 0, 0, 0, 1, 0],
    "STRAFE_FORWARD_LEFT": [1, 0, 0, 0, 0, 1],
    "NONE": [0, 0, 0, 0, 0, 0]
}

# ✅ Completely Revised Obstacle Detection and Avoidance
class ObstacleDetector:
    def __init__(self):
        self.last_positions = []
        self.position_history = []
        self.stuck_threshold = 4
        self.min_movement_threshold = 10.0
        self.consecutive_no_movement = 0
        self.escape_strategy_index = 0
        self.current_escape_step = 0
        self.escape_sequence = []
        
        self.escape_strategies = [
            self._create_backup_turn_right_sequence,
            self._create_backup_turn_left_sequence,
            self._create_strafe_right_sequence,
            self._create_strafe_left_sequence,
            self._create_zigzag_sequence,
            self._create_spiral_out_sequence
        ]
    
    def _create_backup_turn_right_sequence(self):
        return [(ACTIONS["MOVE_BACKWARD"], 8), (ACTIONS["TURN_RIGHT"], 8), (ACTIONS["MOVE_FORWARD"], 5)]
    
    def _create_backup_turn_left_sequence(self):
        return [(ACTIONS["MOVE_BACKWARD"], 8), (ACTIONS["TURN_LEFT"], 8), (ACTIONS["MOVE_FORWARD"], 5)]
    
    def _create_strafe_right_sequence(self):
        return [(ACTIONS["STRAFE_RIGHT"], 8), (ACTIONS["MOVE_FORWARD"], 5), (ACTIONS["STRAFE_LEFT"], 4)]
    
    def _create_strafe_left_sequence(self):
        return [(ACTIONS["STRAFE_LEFT"], 8), (ACTIONS["MOVE_FORWARD"], 5), (ACTIONS["STRAFE_RIGHT"], 4)]
    
    def _create_zigzag_sequence(self):
        return [
            (ACTIONS["TURN_RIGHT"], 5), (ACTIONS["MOVE_FORWARD"], 3),
            (ACTIONS["TURN_LEFT"], 10), (ACTIONS["MOVE_FORWARD"], 3),
            (ACTIONS["TURN_RIGHT"], 5), (ACTIONS["MOVE_FORWARD"], 5)
        ]
    
    def _create_spiral_out_sequence(self):
        sequence = []
        for i in range(1, 4):
            sequence.extend([
                (ACTIONS["MOVE_FORWARD"], i * 3), (ACTIONS["TURN_RIGHT"], 8),
                (ACTIONS["MOVE_FORWARD"], i * 3), (ACTIONS["TURN_RIGHT"], 8),
                (ACTIONS["MOVE_FORWARD"], i * 3), (ACTIONS["TURN_RIGHT"], 8),
                (ACTIONS["MOVE_FORWARD"], i * 3), (ACTIONS["TURN_RIGHT"], 8)
            ])
        return sequence
    
    def record_position(self, position):
        self.position_history.append(position)
        if len(self.position_history) > 20:
            self.position_history = self.position_history[-20:]
            
        if len(self.position_history) < 4:
            return False
            
        recent_positions = self.position_history[-4:]
        distances = [math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    for p1, p2 in zip(recent_positions, recent_positions[1:])]
        total_distance = sum(distances)
        
        if total_distance < self.min_movement_threshold:
            self.consecutive_no_movement += 1
            if self.consecutive_no_movement >= self.stuck_threshold:
                print(f"🚨 STUCK DETECTED: Minimal movement ({total_distance:.2f}) over {self.consecutive_no_movement} checks")
                return True
        else:
            self.consecutive_no_movement = 0
            
        return False
    
    def get_escape_action(self):
        if not self.escape_sequence:
            strategy_func = self.escape_strategies[self.escape_strategy_index]
            self.escape_sequence = strategy_func()
            self.escape_strategy_index = (self.escape_strategy_index + 1) % len(self.escape_strategies)
            self.current_escape_step = 0
            print(f"🚑 Starting new escape strategy {self.escape_strategy_index}")
        
        action, duration = self.escape_sequence[self.current_escape_step]
        self.current_escape_step += 1
        
        if self.current_escape_step >= len(self.escape_sequence):
            print("✅ Completed escape sequence")
            self.escape_sequence = []
            self.current_escape_step = 0
            self.consecutive_no_movement = 0
        
        return action, duration

# ✅ Create Path with Dynamic Waypoint Density
def process_path(path, min_points=10, max_points=20):
    if not path:
        return []
        
    if len(path) <= min_points:
        return path
    
    target_points = min(max_points, max(min_points, len(path) // 15))
    processed_path = [path[0]]
    
    complexity_points = []
    for i in range(1, len(path) - 1):
        prev_point = path[i-1]
        curr_point = path[i]
        next_point = path[i+1]
        
        dx1, dy1 = curr_point[0] - prev_point[0], curr_point[1] - prev_point[1]
        dx2, dy2 = next_point[0] - curr_point[0], next_point[1] - curr_point[1]
        
        len1 = max(0.001, math.sqrt(dx1*dx1 + dy1*dy1))
        len2 = max(0.001, math.sqrt(dx2*dx2 + dy2*dy2))
        
        dx1, dy1 = dx1/len1, dy1/len1
        dx2, dy2 = dx2/len2, dy2/len2
        
        direction_change = 1 - (dx1*dx2 + dy1*dy2)
        if direction_change > 0.2:
            complexity_points.append((i, direction_change))
    
    complexity_points.sort(key=lambda x: x[1], reverse=True)
    critical_indices = [i for i, _ in complexity_points[:target_points-2]]
    critical_indices.sort()
    
    if len(critical_indices) < target_points - 2:
        remaining_points = target_points - 2 - len(critical_indices)
        step = max(1, len(path) // (remaining_points + 1))
        regular_indices = [i for i in range(step, len(path) - 1, step)
                         if i not in critical_indices][:remaining_points]
        all_indices = sorted(critical_indices + regular_indices)
    else:
        all_indices = critical_indices[:target_points-2]
    
    for idx in all_indices:
        processed_path.append(path[idx])
    
    if processed_path[-1] != path[-1]:
        processed_path.append(path[-1])
    
    return processed_path

# ✅ Completely Rewritten Movement Along Path
def move_along_path(game, path, calibration):
    print(f"🚀 Moving agent along path with {len(path)} steps...")
    
    processed_path = process_path(path, min_points=10, max_points=25)
    print(f"🗺️ Processed path to {len(processed_path)} waypoints")
    
    actual_path = []
    obstacle_detector = ObstacleDetector()
    
    total_waypoints = len(processed_path) - 1
    waypoints_reached = 0
    last_successful_position = None
    overall_start_time = time.time()
    overall_timeout = 180
    
    for i in range(len(processed_path) - 1):
        if time.time() - overall_start_time > overall_timeout:
            print("⚠️ Overall timeout reached. Ending navigation.")
            break
            
        target_point = processed_path[i + 1]
        target_x, target_y = convert_coordinates(target_point[0], target_point[1], calibration)
        
        print(f"\n📍 Navigating to waypoint {i+1}/{total_waypoints}:")
        print(f"  - Target: ({target_x:.1f}, {target_y:.1f})")
        
        waypoint_start_time = time.time()
        waypoint_timeout = 30
        reached = False
        min_distance_to_target = float('inf')
        progress_check_time = time.time()
        last_progress_check_distance = float('inf')
        
        while not reached and (time.time() - waypoint_start_time) < waypoint_timeout:
            if time.time() - overall_start_time > overall_timeout:
                print("⚠️ Overall timeout reached during waypoint navigation.")
                break
                
            state = game.get_state()
            if state is None:
                print("⚠️ No state received! Waiting...")
                time.sleep(0.1)
                continue
                
            game_vars = state.game_variables
            if len(game_vars) < 3:
                print(f"⚠️ Warning: Expected 3 game variables, got {len(game_vars)}")
                time.sleep(0.1)
                continue
                
            agent_x, agent_y, agent_angle = game_vars[:3]
            current_position = (agent_x, agent_y)
            
            actual_path.append(current_position)
            
            dx = target_x - agent_x
            dy = target_y - agent_y
            distance = math.sqrt(dx*dx + dy*dy)
            min_distance_to_target = min(min_distance_to_target, distance)
            
            if distance < 60:
                print(f"✅ Reached waypoint {i+1}! (distance: {distance:.1f})")
                reached = True
                waypoints_reached += 1
                last_successful_position = current_position
                break
            
            if time.time() - progress_check_time > 3:
                progress = last_progress_check_distance - distance
                print(f"  Progress check: {progress:.1f} units (distance: {distance:.1f})")
                if progress < 5:
                    print("  ⚠️ Minimal progress detected")
                last_progress_check_distance = distance
                progress_check_time = time.time()
            
            if obstacle_detector.record_position(current_position):
                print(f"🛑 Stuck at ({agent_x:.1f}, {agent_y:.1f}), executing escape maneuver...")
                escape_start = time.time()
                escape_timeout = 10
                
                while time.time() - escape_start < escape_timeout:
                    action, duration = obstacle_detector.get_escape_action()
                    game.make_action(action, duration)
                    state = game.get_state()
                    if state:
                        current_vars = state.game_variables
                        current_x, current_y = current_vars[:2]
                        actual_path.append((current_x, current_y))
                    time.sleep(0.1)
                    if obstacle_detector.current_escape_step == 0:
                        break
                
                progress_check_time = time.time()
                last_progress_check_distance = math.sqrt((target_x - agent_x)**2 + (target_y - agent_y)**2)
                continue
            
            target_angle = math.degrees(math.atan2(dy, dx)) % 360
            agent_angle = agent_angle % 360
            angle_diff = (target_angle - agent_angle + 180) % 360 - 180
            
            if abs(angle_diff) > 15:
                turn_action = ACTIONS["TURN_LEFT"] if angle_diff > 0 else ACTIONS["TURN_RIGHT"]
                turn_tics = min(5, max(2, abs(int(angle_diff / 20))))
                game.make_action(turn_action, turn_tics)
            elif distance > 150:
                game.make_action(ACTIONS["MOVE_FORWARD"], 5)
            elif distance > 80:
                if abs(angle_diff) < 5:
                    game.make_action(ACTIONS["MOVE_FORWARD"], 3)
                elif angle_diff > 0:
                    game.make_action(ACTIONS["FORWARD_LEFT"], 2)
                else:
                    game.make_action(ACTIONS["FORWARD_RIGHT"], 2)
            else:
                if abs(angle_diff) < 5:
                    game.make_action(ACTIONS["MOVE_FORWARD"], 2)
                elif angle_diff > 0:
                    game.make_action(ACTIONS["FORWARD_LEFT"], 1)
                else:
                    game.make_action(ACTIONS["FORWARD_RIGHT"], 1)
            
            time.sleep(0.05)
        
        if not reached:
            print(f"⚠️ Timeout reached for waypoint {i+1}. Best distance: {min_distance_to_target:.1f}")
            if min_distance_to_target < 80:
                print("  - Close enough to continue")
                waypoints_reached += 0.5
            if waypoints_reached / (i + 1) < 0.5 and i > 3:
                print("⚠️ Too many waypoint failures, skipping ahead...")
                i += 2
    
    print(f"\n📊 Navigation Statistics:")
    print(f"  - Waypoints reached: {waypoints_reached}/{total_waypoints}")
    print(f"  - Success rate: {waypoints_reached/total_waypoints*100:.1f}%")
    print(f"  - Total time: {time.time() - overall_start_time:.1f} seconds")
    
    return actual_path

# ✅ Path Visualization
def visualize_paths(binary_map, planned_path, actual_path, calibration):
    plt.figure(figsize=(12, 10))
    
    plt.imshow(binary_map, cmap="gray", origin="upper")
    
    planned_x, planned_y = zip(*[(p[0], p[1]) for p in planned_path])
    plt.plot(planned_x, planned_y, 'r-', linewidth=2, label="A* Path")
    plt.scatter(planned_x[0], planned_y[0], color="yellow", s=100, marker="*", label="Start")
    plt.scatter(planned_x[-1], planned_y[-1], color="blue", s=100, marker="*", label="Goal")
    
    if actual_path and len(actual_path) > 1:
        # Convert ViZDoom coordinates back to map coordinates (approximate inverse)
        scale_x, scale_y, x_offset, y_offset, doom_x_offset, doom_y_offset = calibration
        map_actual_path = []
        for x, y in actual_path:
            map_x = x_offset + (x - doom_x_offset) / scale_x
            map_y = y_offset + (y - doom_y_offset) / scale_y
            map_actual_path.append((map_x, map_y))
        
        actual_x, actual_y = zip(*map_actual_path)
        plt.plot(actual_x, actual_y, 'g-', linewidth=2, label="Agent Path")
    
    plt.title("A* Path on Binary Map")
    plt.legend()
    plt.savefig("path_visualization.png", dpi=300)
    plt.show()

# ✅ Main Execution
def main():
    # Compute A* path
    path = astar(start, goal, binary_map)
    
    if not path:
        print("❌ No path found!")
        return
    
    print(f"✅ Path found: {len(path)} steps")
    
    # Calculate calibration once
    calibration = calibrate_coordinates()
    
    # Initialize game
    game = initialize_game()
    time.sleep(2)  # Give the game a moment to initialize fully
    
    # Get initial position for verification
    state = game.get_state()
    if state and len(state.game_variables) >= 2:
        init_x, init_y = state.game_variables[:2]
        print(f"\n📌 Initial position: ({init_x:.1f}, {init_y:.1f})")
        
        # Verify coordinate conversion accuracy
        expected_start = convert_coordinates(start[0], start[1], calibration)
        print(f"   Expected spawn position: {expected_start}")
        
        # Calculate spawn point discrepancy
        spawn_distance = math.sqrt((init_x - expected_start[0])**2 + (init_y - expected_start[1])**2)
        if spawn_distance > 50:
            print(f"⚠️ WARNING: Coordinate conversion may be incorrect!")
            print(f"   Distance between expected and actual spawn: {spawn_distance:.1f}")
            # Dynamic recalibration (simple adjustment)
            scale_x, scale_y, x_offset, y_offset, _, _ = calibration
            doom_x_offset = init_x - (start[0] - x_offset) * scale_x
            doom_y_offset = init_y - (start[1] - y_offset) * scale_y
            calibration = (scale_x, scale_y, x_offset, y_offset, doom_x_offset, doom_y_offset)
            print(f"   Recalibrated Doom Offsets: ({doom_x_offset:.1f}, {doom_y_offset:.1f})")
    
    # Move along path
    actual_path = move_along_path(game, path, calibration)
    
    # Visualize paths
    visualize_paths(binary_map, path, actual_path, calibration)
    
    game.close()

if __name__ == "__main__":
    main()