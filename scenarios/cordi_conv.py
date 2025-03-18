import vizdoom as vzd

# Step 1: Initialize the DoomGame object
game = vzd.DoomGame()

# Step 2: Load the configuration file
# Assuming 'basic.cfg' is in the current directory; adjust the path if needed
game.load_config("basic.cfg")

# Step 3: Set up the game
# The mode is already set to PLAYER in the config, and window_visible is true
game.init()

# Step 4: Start a new episode to place the player at the spawn point
game.new_episode()

# Step 5: Get the initial game state
state = game.get_state()

# Step 6: Extract and display the spawn point coordinates
if state is not None:
    game_vars = state.game_variables
    if len(game_vars) >= 2:  # Ensure POSITION_X and POSITION_Y are available
        spawn_x = game_vars[0]  # POSITION_X
        spawn_y = game_vars[1]  # POSITION_Y
        print(f"Spawn point coordinates: ({spawn_x}, {spawn_y})")
    else:
        print("Error: Game variables do not include enough position data.")
else:
    print("Error: No game state available.")