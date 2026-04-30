# Screen Settings
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 60

# Colors (R, G, B)
# Pygame uses tuples of three numbers (0-255) for Red, Green, and Blue
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (100, 100, 100) # For our track
GREEN = (0, 153, 0)     # For our "Mud" or grass
RED   = (200, 0, 0)     # For the car (initially)

# Car Physics
CAR_ACCELERATION = 0.1  # Much lower than 0.5
CAR_FRICTION = 0.99     # Closer to 1.0 means less "drag," so it glides longer
CAR_ROTATION_SPEED = 3  # Slightly slower turning makes lines more important
CAR_GRIP = 0.1

# Penalty Settings
GRASS_FRICTION = 0.93 # Standard was 0.99. Lower number = more drag.
GRASS_ACCEL_LIMIT = 0.9 # Reduce acceleration power by 70% in mud
