import pybullet as p
import pybullet_data
import time
import random
import cv2
import numpy as np
from robot import Robot
# Initializing pybullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(enableConeFriction=1, contactBreakingThreshold=0.001)
p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
p.setPhysicsEngineParameter(enableFileCaching=1)
p.setPhysicsEngineParameter(solverResidualThreshold=0.0001)
p.setGravity(0, 0, -30.8)
# to load ground
plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.3, 0.0, 1])  # Brown color
p.changeVisualShape(plane_id, -1, textureUniqueId=-1)  # Disable texture
arena_length = 10
arena_width = 8
wall_thickness = 0.1
wall_height = 0.0
safe_zone_radius = 2  
wall_positions = [
    (-arena_length / 2, 0, wall_height / 2),
    (arena_length / 2, 0, wall_height / 2),
    (0, -arena_width / 2, wall_height / 2),
    (0, arena_width / 2, wall_height / 2)
]
wall_sizes = [
    [wall_thickness, arena_width / 2, wall_height],
    [wall_thickness, arena_width / 2, wall_height],
    [arena_length / 2, wall_thickness, wall_height],
    [arena_length / 2, wall_thickness, wall_height]
]
# to Create Walls
walls = []
wall_friction = 1.0  # Increase friction to ensure no unintended movement
# Change the wall color to brown
brown_color =  [1.0, 0.5, 0.0, 1]  # RGBA for brown

COLOR_TOKENS = {
    "Wet Waste": 1,   # Blue
    "Dry Waste": 2,   # Green
    "Hazardous": 3   # Red
}
cylinder_colors = {
    "Wet Waste": [0, 0, 1, 1],    # Blue
    "Dry Waste": [0, 1, 0, 1],    # Green
    "Hazardous": [1, 0, 0, 1]    # Red
}
cylinders = []
num_cylinders_per_type = 10
spawn_positions = []  # Store existing cylinder positions to prevent overlap

def is_valid_spawn(x, y, radius=0.5):
    """Check if a new cylinder position is valid (no overlap, not in bin, not in robot zone)."""
    for px, py in spawn_positions:
        if np.linalg.norm([x - px, y - py]) < radius:
            return False  # Too close to another cylinder
    return True

for waste_type, color in cylinder_colors.items():
    color_token = COLOR_TOKENS[waste_type]
    for i in range(1, num_cylinders_per_type + 1):
        while True:
            x_pos = random.uniform(-arena_length / 2 + 1, arena_length / 2 - 1)
            y_pos = random.uniform(-arena_width / 2 + 1, arena_width / 2 - 1)

            if np.linalg.norm([x_pos, y_pos]) > safe_zone_radius and is_valid_spawn(x_pos, y_pos):
                spawn_positions.append((x_pos, y_pos))  # Store position to prevent future overlap
                break  # Ensure valid placement

        cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
        cylinder_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cylinder_shape, basePosition=[x_pos, y_pos, 0.15])
        p.changeVisualShape(cylinder_body, -1, rgbaColor=color)
        cylinders.append((cylinder_body, color_token, i))

robot = Robot(start_pos=[0, 0, 0.15], cylinders=cylinders, speed=6, camera_enabled=True)

# Simulation Loop
for _ in range(10000):
    robot.update()
    p.stepSimulation()
    time.sleep(1 / 240)

cv2.destroyAllWindows()
p.disconnect()