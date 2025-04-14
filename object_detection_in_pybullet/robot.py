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
    def __init__(self, start_pos, cylinders, speed=0.3, camera_enabled=True):
        self.robot_id = p.loadURDF("r2d2.urdf", start_pos)
        self.available_cylinders = cylinders[:]  
        self.collected_cylinders = []  
        self.base_speed = speed
        self.current_speed = 0.0
        self.acceleration = 0.02
        self.camera_enabled = camera_enabled
        self.attached_object = None
        self.attachment_constraint = None  
        self.target_bin = None  
        self.last_motion_direction = 0  
        self.bin_fill_index = {1: 0, 2: 0, 3: 0}  
        self.tray_fill_index = 0  # Keeps track of tray filling
        self.current_cylinder_id=None
        self.paused = False
        self.model = load_model('GarbageCollectorCNN.h5')
        p.changeDynamics(self.robot_id, -1, mass=1000)
        # to Create a tray behind the robot
        self.tray_id, self.partition_ids = self.create_open_tray(start_pos)

        p.changeDynamics(self.robot_id, -1, linearDamping=0, angularDamping=0)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        if self.camera_enabled:
            self.setup_camera()
    
    def classify_cylinder(self, image):
        """
        To Classify the cylinder using the CNN model.
        
        :param image: The RGB image of the cylinder.
        :return: Predicted class of the cylinder.
                 B G R == 0 1 2
        """
        resized_image = cv2.resize(image, (32, 32))
        normalized_image = resized_image / 255.0
        input_array = img_to_array(normalized_image)
        input_array = input_array.reshape((1, 32, 32, 3))
        prediction = self.model.predict(input_array)
        predicted_class = prediction.argmax(axis=-1)[0]  
        print(f"Predicted class: {predicted_class}")
        return predicted_class

    def take_screenshot_and_classify(self, folder='data_for_training'):
        """
        To Capture the current camera feed, save it as an image, and classify it.
        
        :param folder: Directory where screenshots will be stored.
        :return: Predicted class of the cylinder.
        """
        image = self.get_camera_feed()
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = f"screenshot_{int(time.time())}.png"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, bgr_image)
        print("Screenshot saved to:", filepath)
        predicted_class = self.classify_cylinder(image)
        return predicted_class

    def create_open_tray(self, start_pos):
        """Loads the 'clear_box' URDF as a tray behind the robot."""
        models = models_data.model_lib()
        tray_model = "clear_box"
        tray_scale = 4
        partition_width = 2
        partition_positions = [
            [start_pos[0], start_pos[1] - partition_width, start_pos[2] + 1.1],  # Left (Red)
            [start_pos[0], start_pos[1], start_pos[2] + 1.1],  # Center (Green)
            [start_pos[0], start_pos[1] + partition_width, start_pos[2] + 1.1]   # Right (Blue)
        ]
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  
        partition_ids = []
        i = -1.5
        for pos, color in zip(partition_positions, colors):
            part_id = p.loadURDF(models[tray_model], pos, globalScaling=tray_scale) 
            p.changeVisualShape(part_id, -1, rgbaColor=color)
            partition_ids.append(part_id)
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
            i+= 1.5

        print("Tray with three partitions (R, G, B) attached behind the robot.")
        return partition_ids[0], partition_ids

    def setup_camera(self):
        """
        Set up the camera parameters.
        """
        self.camera_width = 160
        self.camera_height = 80
        self.fov = 30
        self.aspect = self.camera_width / self.camera_height
        self.near_val = 0.02
        self.far_val = 3.0

    def get_camera_feed(self, tilt_factor=0.4):
        """
        Capture the camera image and return it as an RGB array.
        :param tilt_factor: Tilt factor for the camera.
        :return: RGB image array.
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

        img_arr = p.getCameraImage(self.camera_width, self.camera_height, view_matrix, proj_matrix)
        rgb_array = np.reshape(img_arr[2], (self.camera_height, self.camera_width, 4))

        return rgb_array[:, :, :3]

    def detect_nearest_object(self):
        """
        Detects the nearest cylinder to the robot.
        :return: The nearest cylinder ID and its position.
        """
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
        """
        Move the robot towards the nearest cylinder.
        If the robot is close enough, it will pick up the cylinder.
        """
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
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
                #############
                if pred_class == 0:
                    tray_position = [robot_pos[0],robot_pos[1]+1,robot_pos[2]+1]
                elif pred_class == 1:
                    tray_position = [robot_pos[0],robot_pos[1],robot_pos[2]+1]
                elif pred_class == 2:
                    tray_position = [robot_pos[0],robot_pos[1]-1,robot_pos[2]+1]
                #############
                # Attach the cylinder to the tray
                p.resetBasePositionAndOrientation(cylinder_id, tray_position, [0, 0, 0, 0.1])
                self.collected_cylinders.append((cylinder_id, token))    
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                print(f"Collected cylinder {cylinder_id}, token {token}")

            else:  # Move towards the cylinder
                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)
                velocity = self.current_speed * direction
                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0], [0, 0, 0])
                print("Moving to collect cylinder...")

    def check_and_rebalance(self):
        """
        Check if the robot is tilted and rebalance it if necessary.
        """
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        if abs(roll) > 0.2 or abs(pitch) > 0.2:
            #print("Robot tilted! Resetting orientation...")
            upright_orientation = p.getQuaternionFromEuler([0, 0, yaw])
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
        img_arr = p.getCameraImage(self.camera_width, self.camera_height, view_matrix, proj_matrix)
        depth_buffer = np.reshape(img_arr[3], (self.camera_height, self.camera_width))
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
        rgb_image = self.get_camera_feed()      # shape: (height, width, 3)
        depth_image = self.get_camera_depth()     # shape: (height, width)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
        non_brown_mask = cv2.bitwise_not(brown_mask)
        h, w = depth_image.shape
        roi_mask = np.zeros_like(non_brown_mask)
        roi_mask[h//3:2*h//3, w//3:2*w//3] = 255
        final_mask = cv2.bitwise_and(non_brown_mask, roi_mask)
        area = cv2.countNonZero(final_mask)
        # Get the depth values for the non-brown pixels within the ROI.
        depth_values = depth_image[final_mask > 0]
        if depth_values.size == 0:
            #print("No non-brown pixels found in ROI.")
            return False, None, 0
        min_depth = np.min(depth_values)
        #print(f"Non-brown detection: area = {area} pixels, min depth = {min_depth:.2f} m").
        if min_depth <= threshold_depth and area >= min_area:
            #print("Object detected (non-brown) within threshold.")
            return True, min_depth, area
        else:
            #print("No valid non-brown object detected within threshold.")
            return False, min_depth, area

    def take_screenshot(self, folder='data_for_training'):
        """
        Capture the current camera feed and save it as an image file in the specified folder.
        
        :param folder: Directory where screenshots will be stored.
        """
        image = self.get_camera_feed()
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = f"screenshot_{int(time.time())}.png"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, bgr_image)
        #print("Screenshot saved to:", filepath)

    def update(self):
        if self.paused:
            return
        self.check_and_rebalance()
        detected, _, _ = self.detect_depth_object_ignore_brown(threshold_depth=0.43, min_area=100)
        predicted_class = None
        if detected:
            #self.take_screenshot()
            predicted_class = self.take_screenshot_and_classify() 
            print(f"Detected cylinder class: {predicted_class}")
        self.paused = False
        self.get_camera_feed()
        self.move_toward(pred_class=predicted_class)
