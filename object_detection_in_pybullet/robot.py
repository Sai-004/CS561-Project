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
    def __init__(self, start_pos, cylinders, speed=0.05, camera_enabled=True):
        self.robot_id = p.loadURDF("r2d2.urdf", start_pos)
        self.available_cylinders = cylinders[:]  
        self.collected_cylinders = []  
        self.base_speed = speed
        self.current_speed = 0.0
        self.acceleration = 0.02
        self.camera_enabled = camera_enabled
        self.attached_object = None
        self.attachment_constraint = None  
        self.simulation_ready = False
        self.target_bin = None  
        self.last_motion_direction = 0  
        self.bin_fill_index = {1: 0, 2: 0, 3: 0}  
        self.tray_fill_index = 0  # Keeps track of tray filling
        self.current_cylinder_id=None
        self.paused = False
        self.temp = True
        self.target_set = False
        self.last_position = None
        self.distance_traveled = 0
        self.model = load_model('GarbageCollectorCNN.h5')
        p.changeDynamics(self.robot_id, -1, mass=1000)
        # to Create a tray behind the robot
        self.tray_id, self.partition_ids = self.create_open_tray(start_pos)
        #self.wait_for_simulation_to_initialize()
        self.current_target = None

        p.changeDynamics(self.robot_id, -1, linearDamping=0, angularDamping=0)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        if self.camera_enabled:
            self.setup_camera()
        self.yolo_class_names = ["red","green","blue"]
    
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
        self.camera_width = 320
        self.camera_height = 160
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

    def process_current_image(self):
        """
        Process the current camera image to detect red, green, and blue cylinders.
        Find depths of the bounded boxes and return depth values with their positions.
        
        Returns:
            tuple: (depths, positions) where:
                - depths is an array of depth values for each detected cylinder
                - positions is a list of bounding box positions [x, y, width, height]
        """
        rgb_image, depth_image = self._get_rgbd_image()
        # Detect cylinders using color-based detection (simulating YOLO)
        bounding_boxes = self._detect_cylinders(rgb_image)
        depths = []
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            center_x, center_y = int(x + w/2), int(y + h/2)
            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]
                depths.append(depth)
            else:
                depths.append(None)  
        
        return depths, bounding_boxes

    def _get_rgbd_image(self):
        """
        Get RGB and depth images from the camera (simulating an RGBD sensor).
        
        Returns:
            tuple: (rgb_image, depth_image)
        """
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        camera_eye = np.array(robot_pos) + np.array([
            0.2 * np.cos(self.last_motion_direction),
            0.2 * np.sin(self.last_motion_direction),
            -0.2
        ])
        forward_vector = np.array([
            np.cos(self.last_motion_direction),
            np.sin(self.last_motion_direction),
            -0.4 * 0.7
        ])
        target_pos = camera_eye + forward_vector

        view_matrix = p.computeViewMatrix(
            camera_eye.tolist(), target_pos.tolist(), [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near_val, self.far_val)
        img_arr = p.getCameraImage(
            self.camera_width, self.camera_height, view_matrix, proj_matrix
        )
        rgb_array = np.reshape(
            img_arr[2], (self.camera_height, self.camera_width, 4))[:, :, :3]
        rgb_array = rgb_array.astype(np.uint8)  
        print(f"RGB image shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
        # Extract depth buffer and convert to actual distances
        depth_buffer = np.reshape(
            img_arr[3], [self.camera_height, self.camera_width])
        depth_image = self.far_val * self.near_val / \
            (self.far_val - (self.far_val - self.near_val) * depth_buffer)

        return rgb_array, depth_image
    
    def _detect_cylinders(self, image):
        """
        Detect red, green, and blue cylinders while ignoring walls.
        
        Args:
            image: RGB image from the camera
        
        Returns:
            list: List of bounding boxes [x, y, width, height]
        """
        print(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            print("Converted image to uint8")
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, _, v_channel = cv2.split(hsv_image)
        not_black_mask = cv2.threshold(v_channel, 30, 255, cv2.THRESH_BINARY)[1]
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])

        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        # Create masks for each color
        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Apply the not-black mask
        mask_red = cv2.bitwise_and(mask_red, not_black_mask)
        mask_green = cv2.bitwise_and(mask_green, not_black_mask)
        mask_blue = cv2.bitwise_and(mask_blue, not_black_mask)

        # List to store all bounding boxes
        all_bounding_boxes = []

        # Process each color mask
        for mask in [mask_red, mask_green, mask_blue]:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    all_bounding_boxes.append([x, y, w, h])

        return all_bounding_boxes
    
    def find_direction(self, depths, positions):
        """
        Calculate the direction to move based on detected cylinders.
        
        Args:
            depths: List of depth values for each detected cylinder
            positions: List of bounding box positions [x, y, width, height]
        
        Returns:
            float: Direction angle in radians
        """
        if not depths or not positions:
            print("No valid depths or positions")
            return self.last_motion_direction
        
        # Find the closest cylinder (smallest depth)
        closest_index = np.argmin(depths)
        closest_depth = depths[closest_index]
        closest_bbox = positions[closest_index]
        x, y, w, h = closest_bbox
        bbox_center_x = x + w/2
        # Calculate horizontal position relative to image center
        image_center_x = self.camera_width / 2
        relative_x_pos = bbox_center_x - image_center_x
        print(f"Bbox center: {bbox_center_x}, Image center: {image_center_x}")
        print(f"Relative position: {relative_x_pos}")
        # Convert to angle (in radians)
        # Use full FOV for better sensitivity
        angle = (relative_x_pos / image_center_x) * (self.fov * np.pi / 180)
        print(f"Raw angle: {angle:.4f} radians")
        # Apply a scaling factor to make turns more responsive
        scaling_factor = 1.5 #1.5  # Adjust this value to increase/decrease turn sensitivity
        scaled_angle = angle * scaling_factor
        print(f"Scaled angle: {scaled_angle:.4f} radians")
        # Calculate new direction
        direction = self.last_motion_direction + scaled_angle
        print(f"Last motion direction: {self.last_motion_direction:.4f}")
        print(f"New direction: {direction:.4f}")
        # Ensure direction stays within [-pi, pi]
        normalized_direction = ((direction + np.pi) % (2 * np.pi)) - np.pi
        print(f"Final normalized direction: {normalized_direction:.4f}")
        
        return normalized_direction

    def move_in_direction(self, direction):
        """
        Move the robot in the specified direction using current speed and acceleration.
        
        Args:
            direction: Direction angle in radians
        """
        self.current_speed = self.base_speed
        # Calculate velocity components based on direction and speed
        vx = self.current_speed * np.cos(direction)
        vy = self.current_speed * np.sin(direction)
        # Apply velocity to robot
        p.resetBaseVelocity(self.robot_id, [vx, vy, 0], [0, 0, 0])
        # Update last motion direction
        self.last_motion_direction = direction

    def rotate(self, angle_degrees=30):
        """
        Rotate the robot by the specified angle in degrees while staying in place.
        
        Args:
            angle_degrees: Rotation angle in degrees (default: 30)
        """
        angle_radians = angle_degrees * np.pi / 180
        self.last_motion_direction += angle_radians
        self.last_motion_direction = ((self.last_motion_direction + np.pi) % (2 * np.pi)) - np.pi
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        new_orientation = p.getQuaternionFromEuler([0, 0, self.last_motion_direction])
        p.resetBasePositionAndOrientation(self.robot_id, position, new_orientation)
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        
        print(f"Rotated by {angle_degrees} degrees, new direction: {self.last_motion_direction:.2f} radians")

    
    def update(self):
        #if self.temp==False:
            #time.sleep(1)
            #return
        if self.paused:
            return
        self.check_and_rebalance()
        depths, positions = self.process_current_image()
        # self.check_and_rebalance()
        current_position, _ = p.getBasePositionAndOrientation(self.robot_id)
        #print("Depths:", depths)
        # self.check_and_rebalance()
        #print("Positions:", positions)
        #print(self.target_set)
        if depths and positions and self.target_set==False:
            self.simulation_ready = True
            closest_index = np.argmin(depths)
            closest_depth = depths[closest_index]
            closest_bbox = positions[closest_index]
            self.target_depth = closest_depth * 65
            print("@@@@",self.target_depth)
            self.target_set = True

            direction = self.find_direction(depths, positions)
            # self.check_and_rebalance()
            print("----==================",depths)
            print("--------------------------------",positions)
            self.move_in_direction(direction)
            # self.check_and_rebalance()
            self.last_position = current_position
            print(f"Moving in direction: {direction} radians")
        elif not depths and self.target_set==False:
            if self.simulation_ready != False:
                time.sleep(10)
                # self.check_and_rebalance()
                self.rotate(30)
                # self.check_and_rebalance()
                time.sleep(10)
        elif self.target_set==True and self.last_position is not None:
            distance_moved = np.linalg.norm(np.array(current_position[:2]) - np.array(self.last_position[:2]))
            print("----------")
            self.distance_traveled += distance_moved
            print("---------",self.distance_traveled)
            target_travel_distance = 0.8 * self.target_depth
            if self.distance_traveled >= target_travel_distance:
                # self.check_and_rebalance()
                p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
                # self.check_and_rebalance()
                self.current_speed = 0
                self.target_set = False
                self.temp = False
                print(f"Stopped after traveling {self.distance_traveled:.2f}m, reached target distance")
                # self.check_and_rebalance()
                nearest_cylinder, distance = self.detect_nearest_object()
                # self.check_and_rebalance()
                cylinder_id, token, cylinder_pos = nearest_cylinder
                self.distance_traveled = 0
                pred_class = (token-1)%3
                #############
                if pred_class == 0:
                    tray_position = [current_position[0],current_position[1]+1,current_position[2]+1]
                elif pred_class == 1:
                    tray_position = [current_position[0],current_position[1],current_position[2]+1]
                elif pred_class == 2:
                    tray_position = [current_position[0],current_position[1]-1,current_position[2]+1]
                #############
                # Attach the cylinder to the tray
                time.sleep(2)
                print(",,,,,",cylinder_id)
                print("......",tray_position)
                p.resetBasePositionAndOrientation(cylinder_id, tray_position, [0, 0, 0, 0.1])
                self.collected_cylinders.append((cylinder_id, token))    
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                print(f"Collected cylinder {cylinder_id}, token {token}")
            else:
                self.current_speed = self.base_speed
                vx = self.current_speed * np.cos(self.last_motion_direction)
                vy = self.current_speed * np.sin(self.last_motion_direction)
                # self.check_and_rebalance()
                p.resetBaseVelocity(self.robot_id, [vx, vy, 0], [0, 0, 0])
            # self.check_and_rebalance()
        self.paused = False