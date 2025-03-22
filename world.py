import pybullet as p
import pybullet_data
import time
import random
import cv2
import numpy as np
from robot import Robot

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Create Ground Plane
plane_id = p.loadURDF("plane.urdf")

# Arena Dimensions
arena_length = 10
arena_width = 8
wall_thickness = 0.1
wall_height = 0.7
safe_zone_radius = 2  # Robot spawn zone

# Create Walls
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

walls = []
for pos, size in zip(wall_positions, wall_sizes):
    wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    wall_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, basePosition=pos)
    walls.append(wall_body)

# Color Tokens
COLOR_TOKENS = {
    "Wet Waste": 1,   # Blue
    "Dry Waste": 2,   # Green
    "Hazardous": 3,   # Red
    "Not Waste": 4    # White
}

# Create Bins (Each bin gets a color token)
bin_positions = [
    (-arena_length / 2 + 1, -arena_width / 2 + 1),  # Blue Bin
    (arena_length / 2 - 1, -arena_width / 2 + 1),  # Green Bin
    (-arena_length / 2 + 1, arena_width / 2 - 1)   # Red Bin
]

bin_colors = [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]]  # Blue, Green, Red
bin_tokens = [1, 2, 3]  # Matching color tokens


bins = []
for pos, color, token in zip(bin_positions, bin_colors, bin_tokens):
    bin_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.2])
    bin_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bin_shape, basePosition=[pos[0], pos[1], 0.1])
    p.changeVisualShape(bin_body, -1, rgbaColor=color)
    bins.append((bin_body, token, pos))  # Store (bin ID, bin token, bin position)


# Create Cylinders (Each cylinder gets a color token + unique cylinder number)
cylinder_colors = {
    "Wet Waste": [0, 0, 1, 1],    # Blue
    "Dry Waste": [0, 1, 0, 1],    # Green
    "Hazardous": [1, 0, 0, 1],    # Red
    "Not Waste": [1, 1, 1, 1]     # White
}

cylinders = []
num_cylinders_per_type = 10
for waste_type, color in cylinder_colors.items():
    color_token = COLOR_TOKENS[waste_type]

    for i in range(1, num_cylinders_per_type + 1):
        while True:
            x_pos = random.uniform(-arena_length / 2 + 1, arena_length / 2 - 1)
            y_pos = random.uniform(-arena_width / 2 + 1, arena_width / 2 - 1)
            if np.linalg.norm([x_pos, y_pos]) > safe_zone_radius:
                break  # Ensure cylinders do not spawn in the robot's initial area

        cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=0.3)
        cylinder_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cylinder_shape, basePosition=[x_pos, y_pos, 0.15])
        p.changeVisualShape(cylinder_body, -1, rgbaColor=color)
        
        # Assign tokens: (cylinder ID, color token, cylinder number)
        cylinders.append((cylinder_body, color_token, i))

# Create Robot
robot = Robot(start_pos=[0, 0, 0.15], cylinders=cylinders, bins=bins, speed=6, camera_enabled=True)

# Simulation Loop
for _ in range(5000):
    robot.update()
    p.stepSimulation()
    
    if robot.camera_enabled:
        camera_feed = robot.get_camera_feed()
        if camera_feed is not None:
            cv2.imshow("Robot View", camera_feed)
            cv2.waitKey(1)
    
    time.sleep(1 / 240)

cv2.destroyAllWindows()
p.disconnect()
