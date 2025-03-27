import pybullet as p
import pybullet_data
import time
import random
import cv2
import numpy as np
from robot import Robot
from pybullet_URDF_models.urdf_models import models_data
from pybullet_object_models.pybullet_object_models import ycb_objects
import os

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(enableConeFriction=1, contactBreakingThreshold=0.001)
p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
# p.setPhysicsEngineParameter(enableFileCaching=0)
p.setPhysicsEngineParameter(solverResidualThreshold=0.0001)

p.setGravity(0, 0, -30.8)


# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Hide grid & UI elements

# Load Ground Plane
plane_id = p.loadURDF("plane.urdf")

# Set the floor color to brown
p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.3, 0.0, 1])  # Brown color
p.changeVisualShape(plane_id, -1, textureUniqueId=-1)  # Disable texture
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

# Create Walls (Fixed)
walls = []
wall_friction = 1.0  # Increase friction to ensure no unintended movement

for pos, size in zip(wall_positions, wall_sizes):
    wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)

    # Create a static (immovable) wall
    wall_body = p.createMultiBody(
        baseMass=0,  # Zero mass → Fixed in place
        baseCollisionShapeIndex=wall_shape,
        basePosition=pos
    )

    # Set high friction to prevent unintended sliding
    p.changeDynamics(wall_body, -1, lateralFriction=wall_friction)

    walls.append(wall_body)

# Color Tokens
COLOR_TOKENS = {
    "Wet Waste": 1,   # Blue
    "Dry Waste": 2,   # Green
    "Hazardous": 3,   # Red
    "Not Waste": 4    # White
}

# Create Bins (LARGER SIZE)
bin_positions = [
    (-arena_length / 2 + 1, arena_width / 2 - 1),   # Blue Bin (Top)
    (-arena_length / 2 + 1, 0),                     # Green Bin (Middle)
    (-arena_length / 2 + 1, -arena_width / 2 + 1)   # Red Bin (Bottom)
]

bin_colors = [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]]  # Blue, Green, Red
bin_tokens = [1, 2, 3]  # Matching color tokens

bins = []
bin_areas = []  # Store bin areas to prevent cylinder spawning inside them
bin_radius = 1.0  # Safe distance from bins

for pos, color, token in zip(bin_positions, bin_colors, bin_tokens):
    bin_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.3])  # INCREASED SIZE
    bin_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bin_shape, basePosition=[pos[0], pos[1], 0.15])
    p.changeVisualShape(bin_body, -1, rgbaColor=color)
    bins.append((bin_body, token, pos))
    bin_areas.append(pos)  # Store bin position

    # # ✅ Create a small sphere at the bin position (for debugging)
    # marker_radius = 0.5  # Small sphere size
    # marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=marker_radius, rgbaColor=[1, 0, 0, 1])  # Red sphere
    # marker_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=marker_radius)  # Collision shape

    # marker_id = p.createMultiBody(baseMass=0, 
    #                               baseCollisionShapeIndex=marker_collision, 
    #                               baseVisualShapeIndex=marker_visual, 
    #                               basePosition=[pos[0]+3, pos[1]-0.1, 1])  # Slightly above ground


# Create Cylinders (Ensure No Overlapping & Avoid Bins)
cylinder_colors = {
    "Wet Waste": [0, 0, 1, 1],    # Blue
    "Dry Waste": [0, 1, 0, 1],    # Green
    "Hazardous": [1, 0, 0, 1],    # Red
    "Not Waste": [1, 1, 1, 1]     # White
}


p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Ensure search path is set
# Load Waste Models
models = models_data.model_lib()
waste_categories = {
    "Dry Waste": ['plate', 'spoon', 'potato_chip_1', 'sugar_box', 'fork', 'potato_chip_2', 'orion_pie'],
    "Wet Waste": ['glue_1', 'shampoo', 'plastic_banana', 'plastic_orange', 'toothpaste_1', 'correction_fluid'],
    "Hazardous": ['bleach_cleanser', 'cleanser', 'repellent', 'cracker_box', 'remote_controller_2'],
    "Not Waste": ['mug', 'remote_controller_1', 'scissors', 'power_drill', 'green_cup', 'book_1']
}
###############
def get_path(_model_name: str) -> str:
    return os.path.join(ycb_objects.getDataPath(), _model_name, "model.urdf")

ycb_waste_categories = {
    "Dry Waste": ['YcbChipsCan', 'YcbCrackerBox', 'YcbGelatinBox', 'YcbMasterChefCan'],
    "Wet Waste": ['YcbPottedMeatCan', 'YcbFoamBrick', 'YcbMustardBottle','YcbTomatoSoupCan'],
    "Hazardous": ['YcbScissors', 'YcbHammer', 'YcbMediumClamp','YcbTennisBall'],
    "Not Waste": ['YcbBanana','YcbPear','YcbStrawberry']
}
###############
# Object Spawning Optimization
cylinders = []
num_objects_per_type = 5  # 10 per category (Total: 40 objects)
spawn_positions = []

def is_valid_spawn(x, y, radius=0.5):
    """Ensures no overlap with existing objects and avoids bins."""
    for px, py in spawn_positions:
        if np.linalg.norm([x - px, y - py]) < radius:
            return False
    for bx, by in bin_areas:
        if np.linalg.norm([x - bx, y - by]) < 2 * bin_radius:
            return False
    return True

# Create objects (ONLY visual models from URDF, NO extra physics shapes)
# for waste_type, model_list in waste_categories.items():
#     for i in range(num_objects_per_type):
#         while True:
#             x_pos = random.uniform(-arena_length / 2 + 1, arena_length / 2 - 1)
#             y_pos = random.uniform(-arena_width / 2 + 1, arena_width / 2 - 1)
#             if np.linalg.norm([x_pos, y_pos]) > safe_zone_radius and is_valid_spawn(x_pos, y_pos):
#                 spawn_positions.append((x_pos, y_pos))
#                 break  # Ensure valid placement

#         model_name = random.choice(model_list)  # Choose random model
#         model_id = p.loadURDF(models[model_name], [x_pos, y_pos, 0.15])  # Load visual only
#         cylinders.append((model_id, waste_type, i))
#####################
for waste_type, model_list in ycb_waste_categories.items():
    for i in range(num_objects_per_type):
        while True:
            x_pos = random.uniform(-arena_length / 2 + 1, arena_length / 2 - 1)
            y_pos = random.uniform(-arena_width / 2 + 1, arena_width / 2 - 1)
            if np.linalg.norm([x_pos, y_pos]) > safe_zone_radius and is_valid_spawn(x_pos, y_pos):
                spawn_positions.append((x_pos, y_pos))
                break  # Ensure valid placement

        model_name = random.choice(model_list)  # Choose random model
        model_id = p.loadURDF(get_path(model_name), [x_pos, y_pos, 0.15],flags=p.URDF_USE_INERTIA_FROM_FILE)  # Load visual only
        cylinders.append((model_id, waste_type, i))
#####################
# Create Robot
robot = Robot(start_pos=[0, 0, 0.15], cylinders=cylinders, bins=bins, speed=12, camera_enabled=True)

# Reduce number of iterations per render update
frame_skip = 6  # Step simulation multiple times before rendering

# Simulation Loop
for _ in range(5000):
    robot.update()
    p.stepSimulation()
    
    if robot.camera_enabled:
        camera_feed = robot.get_camera_feed()
        if camera_feed is not None:
            camera_feed = camera_feed.astype("uint8")
            cv2.imshow("Robot View", camera_feed)
            cv2.waitKey(1)
    
    time.sleep(1 / 240)

cv2.destroyAllWindows()
p.disconnect()