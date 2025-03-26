import os
import time
import pybullet as p
import pybullet_data
from urdf_models import models_data
import random

# initialize the GUI and others
p.connect(p.GUI)
p.resetDebugVisualizerCamera(3, 90, -30, [0.0, -0.0, -0.0])
p.setTimeStep(1 / 240.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load urdf data
models = models_data.model_lib()

# load model list
namelist = models.model_name_list
# print("Look at what we have {}".format(namelist))

# Load table and plane
p.loadURDF("plane.urdf")

flags = p.URDF_USE_INERTIA_FROM_FILE
# randomly get a model

random_model = namelist[random.randint(0, len(namelist) - 1)]
random_model= "clear_box"
print(random_model)
p.loadURDF(models[random_model], [0., 0., 0.8 + 0.15], flags=flags)

p.setGravity(0, 0, -9.8)

while 1:
    p.stepSimulation()
    time.sleep(1./240)