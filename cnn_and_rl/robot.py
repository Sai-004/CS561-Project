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
import signal
import atexit

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class RLAgent: 
    def __init__(self, q_table_path='q_table.npy', 
                 num_states=3, num_actions=3,
                 learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.1):
        self.q_table_path = q_table_path
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.save_interval = 1  # Save every 50 updates
        self.update_count = 0
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
        # Try loading existing Q-table
        if os.path.exists(self.q_table_path):
            print(f"Loading Q-table from {self.q_table_path}")
            self.q_table = np.load(self.q_table_path)
            # Handle potential dimension mismatches
            if self.q_table.shape != (num_states, num_actions):
                print("Warning: Existing Q-table dimensions mismatch. Creating new one.")
                self.q_table = np.zeros((num_states, num_actions))
        else:
            print("No existing Q-table found. Initializing new one.")
            self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward):
        old_value = self.q_table[state, action]
        best_future = np.max(self.q_table[state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * best_future)
        self.q_table[state, action] = new_value
        self.update_count += 1
        print("Outside saved interval")
        if self.update_count % self.save_interval == 0:
            print("Inside save interval")
            self.save_q_table()
        

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, saving Q-table...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        self.save_q_table()
        print("Cleanup completed. Q-table saved.")

    def save_q_table(self):
        try:
            np.save(self.q_table_path , self.q_table)
            os.replace(self.q_table_path , self.q_table_path)
            print(f"Q-table safely saved to {self.q_table_path}")
        except Exception as e:
            print(f"Error saving Q-table: {str(e)}")

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
        self.tray_fill_index = 0  
        self.current_cylinder_id=None
        self.paused = False
        
        self.rl_agent = RLAgent(
            q_table_path='cylinder_q_table.npy',
            num_states=3,  
            num_actions=3  
        )
        # Load the CNN model
        self.model = load_model('GarbageCollectorCNN.h5') 
        p.changeDynamics(self.robot_id, -1, mass=1000)
        self.tray_id, self.partition_ids = self.create_open_tray(start_pos)
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
        resized_image = cv2.resize(image, (32, 32))
        normalized_image = resized_image / 255.0
        input_array = img_to_array(normalized_image)
        input_array = input_array.reshape((1, 32, 32, 3))  # Add batch dimension
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
        tray_scale = 4 # Adjusted size
        partition_width = 2
        # Adjust based on tray size
        partition_positions = [
            [start_pos[0], start_pos[1] - partition_width, start_pos[2] + 1.1],  # Left (Red)
            [start_pos[0], start_pos[1], start_pos[2] + 1.1],  # Center (Green)
            [start_pos[0], start_pos[1] + partition_width, start_pos[2] + 1.1]   # Right (Blue)
        ]
        colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  # RGBA colors
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

        view_matrix = p.computeViewMatrix(
            camera_eye.tolist(), target_pos.tolist(), [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near_val, self.far_val)

        img_arr = p.getCameraImage(
            self.camera_width, self.camera_height, view_matrix, proj_matrix)
        rgb_array = np.reshape(
            img_arr[2], (self.camera_height, self.camera_width, 4))[:, :, :3]
        rgb_array = rgb_array.astype(np.uint8)
        print(f"RGB image shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")

        return rgb_array

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
        nearest_cylinder, distance = self.detect_nearest_object()

        if nearest_cylinder:
            cylinder_id, token, cylinder_pos = nearest_cylinder

            if token == 4:  
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                return

            direction = np.array(cylinder_pos[:2]) - np.array(robot_pos[:2])
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else [0, 0]

            if distance < 0.62:  # If close enough, pick up the cylinder
                p.resetBaseVelocity(self.robot_id, [0, 0, 0])
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
                # if pred_class == 0:
                #     tray_position = [robot_pos[0],robot_pos[1]+1,robot_pos[2]+1]
                # elif pred_class == 1:
                #     tray_position = [robot_pos[0],robot_pos[1],robot_pos[2]+1]
                # elif pred_class == 2:
                #     tray_position = [robot_pos[0],robot_pos[1]-1,robot_pos[2]+1]
                #############

                #############
                # RL-based placement decision
                if pred_class in [0, 1, 2]:
                    action = self.rl_agent.choose_action(pred_class)
                    # Determine tray position based on RL action
                    if action == 0:
                        tray_position = [robot_pos[0], robot_pos[1]+1, robot_pos[2]+1]  # Left partition
                    elif action == 1:
                        tray_position = [robot_pos[0], robot_pos[1], robot_pos[2]+1]    # Center partition
                    else:
                        tray_position = [robot_pos[0], robot_pos[1]-1, robot_pos[2]+1]  # Right partition
                    reward = 1 if action == pred_class else -1
                    self.rl_agent.update(pred_class, action, reward)
                else:
                    # Handle unknown class (W) with default position
                    tray_position = [robot_pos[0], robot_pos[1], robot_pos[2]+1]
                # Attach the cylinder to the tray
                p.resetBasePositionAndOrientation(cylinder_id, tray_position, [0, 0, 0, 0.1])
                self.collected_cylinders.append((cylinder_id, token))    
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                print(f"Collected cylinder {cylinder_id}, token {token}")
            else: 
                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)
                velocity = self.current_speed * direction
                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0], [0, 0, 0])
                print("Moving to collect cylinder...")

    def check_and_rebalance(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        if abs(roll) > 0.2 or abs(pitch) > 0.2:
            print("Robot tilted! Resetting orientation...")
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
        near = self.near_val
        far = self.far_val
        actual_depth = 2.0 * near * far / (far + near - (2.0 * depth_buffer - 1.0) * (far - near))
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
        rgb_image = self.get_camera_feed()  # shape: (height, width, 3)
        depth_image = self.get_camera_depth()  # shape: (height, width)
        # Ensure rgb_image is uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
            print("Converted rgb_image to uint8")
        print(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv_image, lower_brown, upper_brown)
        non_brown_mask = cv2.bitwise_not(brown_mask)
        h, w = depth_image.shape
        roi_mask = np.zeros_like(non_brown_mask)
        roi_mask[h//3:2*h//3, w//3:2*w//3] = 255
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
        print(
            f"Non-brown detection: area = {area} pixels, min depth = {min_depth:.2f} m")
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
        image = self.get_camera_feed()
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Create the folder if it does not exist.
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Generate a unique filename using the current timestamp.
        filename = f"screenshot_{int(time.time())}.png"
        filepath = os.path.join(folder, filename)
        # Save the image
        cv2.imwrite(filepath, bgr_image)

    def update(self):
        if not self.available_cylinders:
            print("All cylinders collected. Terminating program.")
            print("-----",print("After run:", np.load('cylinder_q_table.npy')))
            sys.exit(0)
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
        self.get_camera_feed()
        self.move_toward(pred_class=predicted_class)
