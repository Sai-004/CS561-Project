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

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

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
        self.paused = False

        # Load the CNN model
        self.model = load_model('D:\\Backup_2022-08-01_212503\\8th Semester\\CS561 Artificial Intelligence\\Project\\Koushiks\\KoushikModel.h5')  # Increase robot mass for better stability
        p.changeDynamics(self.robot_id, -1, mass=1000)
        
        # Create a tray behind the robot
        self.tray_id, self.partition_ids = self.create_open_tray(start_pos)

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
    
    def classify_cylinder(self, image):
        """
        Classify the cylinder using the CNN model.
        
        :param image: The RGB image of the cylinder.
        :return: Predicted class of the cylinder.
                 B G R W == 0 1 2 3
        """
        # Resize the image to 32x32 pixels
        resized_image = cv2.resize(image, (32, 32))
        
        # Normalize the image (scale pixel values to [0, 1])
        normalized_image = resized_image / 255.0
        
        # Convert the image to a format suitable for the model
        input_array = img_to_array(normalized_image)
        input_array = input_array.reshape((1, 32, 32, 3))  # Add batch dimension
        
        # Predict the class
        prediction = self.model.predict(input_array)
        predicted_class = prediction.argmax(axis=-1)[0]  # Get the class with the highest probability
        
        print(f"Predicted class: {predicted_class}")
        return predicted_class

    def take_screenshot_and_classify(self, folder='data_for_training'):
        """
        Capture the current camera feed, save it as an image, and classify it.
        
        :param folder: Directory where screenshots will be stored.
        :return: Predicted class of the cylinder.
        """
        # Take a screenshot
        image = self.get_camera_feed()
        
        # Convert the image from RGB to BGR (OpenCV uses BGR)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create the folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Generate a unique filename using the current timestamp
        filename = f"screenshot_{int(time.time())}.png"
        filepath = os.path.join(folder, filename)
        
        # Save the image
        cv2.imwrite(filepath, bgr_image)
        print("Screenshot saved to:", filepath)
        
        # Classify the image
        predicted_class = self.classify_cylinder(image)
        return predicted_class

    def create_open_tray(self, start_pos):
        # # # # """Loads the 'clear_box' URDF as a tray behind the robot."""
       
        # # # # models = models_data.model_lib()

        # # # # tray_model = "clear_box"
        # # # # tray_position = [start_pos[0], start_pos[1], start_pos[2] + 1.2]  # Move behind the robot
        # # # # tray_id = p.loadURDF(models[tray_model], tray_position, globalScaling=15)  # Scale up the tray

        # # # # p.createConstraint(
        # # # #     parentBodyUniqueId=self.robot_id,
        # # # #     parentLinkIndex=-1,
        # # # #     childBodyUniqueId=tray_id,
        # # # #     childLinkIndex=-1,
        # # # #     jointType=p.JOINT_FIXED,
        # # # #     jointAxis=[0, 0, 0],
        # # # #     parentFramePosition=[0, 0, 1.3],  # Properly position it behind
        # # # #     childFramePosition=[0, 0, 0]  # Attach to the tray’s center
        # # # # )

        # # # # print("Tray attached behind the robot using clear_box.")
        # # # # return tray_id
        ######################### MODIF START
        """Loads the 'clear_box' URDF as a tray behind the robot."""
       
        # models = models_data.model_lib()

        # tray_model = "clear_box"
        # tray_position = [start_pos[0], start_pos[1], start_pos[2] + 1.2]  
        # tray_scale = 9
        # tray_id = p.loadURDF(models[tray_model], tray_position, globalScaling=tray_scale)  # Scale up the tray

        # p.createConstraint(
        #     parentBodyUniqueId=self.robot_id,
        #     parentLinkIndex=-1,
        #     childBodyUniqueId=tray_id,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[0, 0, 1.3],  # Properly position it behind
        #     childFramePosition=[0, 0, 0]  # Attach to the tray’s center
        # )

        # print("Tray attached behind the robot using clear_box.")
        # return tray_id
    
        ############################
        models = models_data.model_lib()
        
        tray_model = "clear_box"
        tray_scale = 7  # Adjusted size

        # tray_id = p.loadURDF(models[tray_model], tray_position, globalScaling=tray_scale)

        # p.createConstraint(
        #     parentBodyUniqueId=self.robot_id,
        #     parentLinkIndex=-1,
        #     childBodyUniqueId=tray_id,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[0, 0, 1.0],
        #     childFramePosition=[0, 0, 0]
        # )

        # Creating partitions inside the tray
        partition_width = 1.1
         # Adjust based on tray size
        partition_positions = [
            [start_pos[0], start_pos[1] - partition_width, start_pos[2] + 1.1],  # Left (Red)
            [start_pos[0], start_pos[1], start_pos[2] + 1.1],  # Center (Green)
            [start_pos[0], start_pos[1] + partition_width, start_pos[2] + 1.1]   # Right (Blue)
        ]
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  # RGBA colors
        partition_ids = []
        i = -1.1
        for pos, color in zip(partition_positions, colors):
            part_id = p.loadURDF(models[tray_model], pos, globalScaling=6)  # Smaller boxes
            p.changeVisualShape(part_id, -1, rgbaColor=color)
            partition_ids.append(part_id)

            # Fix partitions inside the tray
            p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=-1,
                childBodyUniqueId=part_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, i, 1.0],
                childFramePosition=[0, 0, 0]
            )
            i+= 1.1

        print("Tray with three partitions (R, G, B) attached behind the robot.")
        return partition_ids[0], partition_ids
        #################### MODIF END

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

    def move_toward(self, pred_class:int = 3):
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
                tray_y = -1*((pred_class-1)) + (row + 1) * cell_spacing
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

    def get_camera_depth(self, tilt_factor=0.4):
        """
        Capture the camera image and return the actual depth map.
        Uses the near and far values to convert the depth buffer into actual depth values.
        """
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

        # p.getCameraImage returns a tuple where index 3 is the depth buffer.
        img_arr = p.getCameraImage(self.camera_width, self.camera_height, view_matrix, proj_matrix)
        depth_buffer = np.reshape(img_arr[3], (self.camera_height, self.camera_width))

        # Convert normalized depth values to actual depth (in meters)
        # Using the formula: actual_depth = far * near / (far - (far-near) * depth_value)
        actual_depth = self.far_val * self.near_val / (self.far_val - (self.far_val - self.near_val) * depth_buffer)
        return actual_depth

    def detect_depth_object_ignore_brown(self, threshold_depth=0.43, min_area=100):
        """
        Detects an object using the depth image but ignores the brown floor.
        First, it filters out brown areas from the RGB camera feed,
        then applies that mask to the depth image in a central ROI.
        
        :param threshold_depth: Distance threshold (in meters) to trigger stopping.
        :param min_area: Minimum number of non-brown pixels with depth <= threshold_depth.
        :return: Tuple (detected_flag, min_depth, area) where detected_flag is True if an object (non-brown)
                is close enough.
        """
        # Obtain RGB and depth images.
        rgb_image = self.get_camera_feed()      # shape: (height, width, 3)
        depth_image = self.get_camera_depth()     # shape: (height, width)
        
        # Convert the RGB image to HSV.
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Define the HSV range for brown (floor) - adjust these values as needed.
        # Typical brown may have hues around 10-20, moderate saturation, and moderate value.
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
        
        # Invert the mask to keep only non-brown regions (potential objects).
        non_brown_mask = cv2.bitwise_not(brown_mask)
        
        # Define a central ROI (for example, the middle third of the image).
        h, w = depth_image.shape
        roi_mask = np.zeros_like(non_brown_mask)
        roi_mask[h//3:2*h//3, w//3:2*w//3] = 255
        
        # Combine the non-brown mask with the ROI mask.
        final_mask = cv2.bitwise_and(non_brown_mask, roi_mask)
        
        # Count the number of non-brown pixels in the ROI.
        area = cv2.countNonZero(final_mask)
        
        # Get the depth values for the non-brown pixels within the ROI.
        depth_values = depth_image[final_mask > 0]
        
        if depth_values.size == 0:
            print("No non-brown pixels found in ROI.")
            return False, None, 0
        
        # Find the minimum depth among these pixels.
        min_depth = np.min(depth_values)
        print(f"Non-brown detection: area = {area} pixels, min depth = {min_depth:.2f} m")
        
        # If a significant area has depth less than or equal to the threshold, consider it an object.
        if min_depth <= threshold_depth and area >= min_area:
            print("Object detected (non-brown) within threshold.")
            return True, min_depth, area
        else:
            print("No valid non-brown object detected within threshold.")
            return False, min_depth, area

    def take_screenshot(self, folder='data_for_training'):
        """
        Capture the current camera feed and save it as an image file in the specified folder.
        
        :param folder: Directory where screenshots will be stored.
        """
        # Retrieve the camera feed (returns an RGB image)
        image = self.get_camera_feed()
        
        # Convert the image from RGB to BGR (OpenCV uses BGR)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create the folder if it does not exist.
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Generate a unique filename using the current timestamp.
        filename = f"screenshot_{int(time.time())}.png"
        filepath = os.path.join(folder, filename)
        
        # Save the image
        cv2.imwrite(filepath, bgr_image)
        #print("Screenshot saved to:", filepath)

    def update(self):
        if self.paused:
            return

        self.check_and_rebalance()
        
        # Use the new depth-based detection ignoring brown areas.
        detected, min_depth, area = self.detect_depth_object_ignore_brown(threshold_depth=0.43, min_area=100)
        predicted_class = None
        if detected:
             # Stop the robot when a non-brown object is detected close enough
            self.take_screenshot()  # Save the screenshot
            predicted_class = self.take_screenshot_and_classify()  # Classify the cylinder
            print(f"Detected cylinder class: {predicted_class}")
        # If no valid object is detected, continue normal operation.
        self.paused = False
        self.get_camera_feed()  # For visualization if needed.
        self.move_toward(pred_class=predicted_class)
