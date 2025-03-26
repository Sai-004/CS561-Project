import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random
from pybullet_URDF_models.urdf_models import models_data
import random
import sys
import os



class Robot:
    def __init__(self, start_pos, cylinders, bins, speed=0.5, camera_enabled=True):
        self.robot_id = p.loadURDF("r2d2.urdf", start_pos)
        self.available_cylinders = cylinders[:]  
        self.collected_cylinders = []  
        self.bins = {token: pos for _, token, pos in bins}  
        self.base_speed = speed
        self.current_speed = 0.0
        self.acceleration = 0.05
        self.camera_enabled = camera_enabled
        self.attached_object = None
        self.attachment_constraint = None  # Store constraint ID separately
        self.target_bin = None  
        self.last_motion_direction = 0  
        self.bin_fill_index = {1: 0, 2: 0, 3: 0}  
        self.tray_fill_index = 0  # Keeps track of tray filling
        self.current_cylinder_id=None

        
        # Increase robot mass for better stability
        p.changeDynamics(self.robot_id, -1, mass=1000)
        
        # Create a tray behind the robot
        self.tray_id = self.create_open_tray(start_pos)

        bin_size = 1  
        self.bin_corners = {}

        for bin_body, token, pos in bins:
            bx, by = pos
            self.bin_corners[token] = [
                (bx - bin_size / 2, by - bin_size / 2),  
                (bx + bin_size / 2, by - bin_size / 2),  
                (bx - bin_size / 2, by + bin_size / 2),  
                (bx + bin_size / 2, by + bin_size / 2)   
            ]

        print("Initialized bin corners:", self.bin_corners)

        p.changeDynamics(self.robot_id, -1, linearDamping=0, angularDamping=0)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        if self.camera_enabled:
            self.setup_camera()

    def create_open_tray(self, start_pos):
        """Loads the 'clear_box' URDF as a tray behind the robot."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Ensure search path is set
        models = models_data.model_lib()

        tray_model = "clear_box"
        tray_position = [start_pos[0], start_pos[1], start_pos[2] + 1.2]  # Move behind the robot
        tray_id = p.loadURDF(models[tray_model], tray_position, globalScaling=15)  # Scale up the tray

        p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=-1,
            childBodyUniqueId=tray_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 1.3],  # Properly position it behind
            childFramePosition=[0, 0, 0]  # Attach to the tray’s center
        )

        print("Tray attached behind the robot using clear_box.")
        return tray_id
    


    def setup_camera(self):
        self.camera_width = 160
        self.camera_height = 80
        self.fov = 30
        self.aspect = self.camera_width / self.camera_height
        self.near_val = 0.02
        self.far_val = 3.0

    def get_camera_feed(self, tilt_factor=0.4):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        velocity, _ = p.getBaseVelocity(self.robot_id)
        velocity_x, velocity_y = velocity[:2]
        speed_magnitude = np.linalg.norm([velocity_x, velocity_y])

        if speed_magnitude > 0.2:
            self.last_motion_direction = np.arctan2(velocity_y, velocity_x)

        camera_eye = np.array(robot_pos) + np.array([
            0.2 * np.cos(self.last_motion_direction),
            0.2 * np.sin(self.last_motion_direction),
            -0.2
        ])
        forward_vector = np.array([
            np.cos(self.last_motion_direction),
            np.sin(self.last_motion_direction),
            -tilt_factor * 0.7
        ])
        target_pos = camera_eye + forward_vector

        view_matrix = p.computeViewMatrix(camera_eye.tolist(), target_pos.tolist(), [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near_val, self.far_val)

        img_arr = p.getCameraImage(self.camera_width, self.camera_height, view_matrix, proj_matrix)
        rgb_array = np.reshape(img_arr[2], (self.camera_height, self.camera_width, 4))

        return rgb_array[:, :, :3]

    def detect_nearest_object(self):
        min_distance = float('inf')
        nearest_cylinder = None

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        for cylinder_id, token, _ in self.available_cylinders:
            cyl_pos, _ = p.getBasePositionAndOrientation(cylinder_id)
            distance = np.linalg.norm(np.array(robot_pos[:2]) - np.array(cyl_pos[:2]))

            if distance < min_distance:
                min_distance = distance
                nearest_cylinder = (cylinder_id, token, cyl_pos)

        return nearest_cylinder, min_distance

    def move_toward(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)

        # 1️⃣ If all cylinders are collected, start placing them in bins
        if not self.available_cylinders and self.collected_cylinders:
            if self.current_cylinder_id is None:
                # Choose a cylinder *only if there's no current one being placed*
                index = random.randint(0, len(self.collected_cylinders) - 1)
                self.current_cylinder_id, self.current_token = self.collected_cylinders.pop(index)
                self.target_bin = self.bins.get(self.current_token, (-4, -2.3))
                print(f"Selected cylinder {self.current_cylinder_id} for bin {self.current_token}")

            bin_height = 2.0  # Adjust this based on your actual bin size

            bin_x, bin_y = self.target_bin  # Get bin center position
            bin_x+=2.5
            bottom_y = bin_y+0.01  # Move toward the bottom edge


            direction = np.array([bin_x, bottom_y]) - np.array(robot_pos[:2])

            distance = np.linalg.norm(direction)

            if distance < 0.1:  # If close enough, place the cylinder
                print(f"Placing cylinder {self.current_cylinder_id} in bin {self.current_token}")
                p.resetBaseVelocity(self.robot_id, [0, 0, 0])

                # 2️⃣ 3x3 Grid Placement inside the Bin
                bin_size = 2.0  
                grid_size = 3  
                cell_spacing = bin_size / (grid_size + 1)

                bx, by = self.target_bin
                row = self.bin_fill_index[self.current_token] // grid_size
                col = self.bin_fill_index[self.current_token] % grid_size

                drop_x = bx - bin_size / 2 + (col + 1) * cell_spacing
                drop_y = by - bin_size / 2 + (row + 1) * cell_spacing
                drop_position = (drop_x, drop_y)

                self.bin_fill_index[self.current_token] += 1  

                # Drop cylinder at the assigned bin position
                p.resetBasePositionAndOrientation(self.current_cylinder_id, [drop_position[0], drop_position[1], 0.9], [0, 0, 0, 1])
                print(f"Placed cylinder {self.current_cylinder_id} in bin {self.current_token}")

                self.current_cylinder_id = None  # ✅ Reset after placement
                self.current_speed = 0.0  

            else:  # Move toward the bin
                direction = direction / np.linalg.norm(direction)
                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)
                velocity = self.current_speed * direction
                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0], [0, 0, 0])
                print("Moving to bin...")

            return  

        # 3️⃣ If there are still cylinders left to collect
        nearest_cylinder, distance = self.detect_nearest_object()

        if nearest_cylinder:
            cylinder_id, token, cylinder_pos = nearest_cylinder

            if token == 4:  # Skip token 4
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                return

            direction = np.array(cylinder_pos[:2]) - np.array(robot_pos[:2])
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else [0, 0]

            if distance < 0.62:  # If close enough, pick up the cylinder
                p.resetBaseVelocity(self.robot_id, [0, 0, 0])

                # 4️⃣ Compute tray placement using a grid system
                tray_size = 2.0
                grid_size = 3
                cell_spacing = tray_size / (grid_size + 1)

                index = self.tray_fill_index % (grid_size * grid_size)  # Loop back after filling 3x3
                row = index // grid_size
                col = index % grid_size

                tray_x = -1 + (col + 1) * cell_spacing
                tray_y = -1 + (row + 1) * cell_spacing
                tray_position = [robot_pos[0] + tray_x, robot_pos[1] + tray_y, robot_pos[2] + 1]

                self.tray_fill_index += 1  

                # Attach the cylinder to the tray
                p.resetBasePositionAndOrientation(cylinder_id, tray_position, [0, 0, 0, 0.2])
                self.collected_cylinders.append((cylinder_id, token))    
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                print(f"Collected cylinder {cylinder_id}, token {token}")

            else:  # Move towards the cylinder
                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)
                velocity = self.current_speed * direction
                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0], [0, 0, 0])
                print("Moving to collect cylinder...")




    def check_and_rebalance(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        if abs(roll) > 0.2 or abs(pitch) > 0.2:  # If tilt exceeds ~15 degrees
            print("Robot tilted! Resetting orientation...")

            # Maintain the current position but reset orientation to upright
            upright_orientation = p.getQuaternionFromEuler([0, 0, yaw])  # Keep yaw, reset roll & pitch
            p.resetBasePositionAndOrientation(self.robot_id, position, upright_orientation)


    def update(self):
        self.check_and_rebalance()
        self.get_camera_feed()
        self.move_toward()
        # self.update_tray_position()