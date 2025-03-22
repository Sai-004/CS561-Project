import pybullet as p
import numpy as np
import cv2
import time

class Robot:
    def __init__(self, start_pos, cylinders, bins, speed=0.5, camera_enabled=True):
        self.robot_id = p.loadURDF("r2d2.urdf", start_pos)
        self.cylinders = cylinders  # List of (cylinder_id, cylinder_color, token)
        self.bins = bins  # List of (bin_id, token)
        self.base_speed = speed
        self.speed = speed  # Current speed, dynamically adjusted
        self.camera_enabled = camera_enabled
        self.attached_object = None  # Track attached object
        self.avoiding_obstacles = False
        self.obstacle_avoidance_end_time = 0

        if self.camera_enabled:
            self.setup_camera()

    def setup_camera(self):
        """Initialize camera parameters and position it slightly in front of the robot and tilted downward."""
        self.camera_width = 160
        self.camera_height = 80
        self.fov = 45  # Increased field of view
        self.aspect = self.camera_width / self.camera_height
        self.near_val = 0.02
        self.far_val = 3.0

    def get_camera_feed(self, tilt_factor=0.4):
        """Captures the robot's camera feed and processes it to detect objects."""
        robot_pos, robot_orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_yaw = p.getEulerFromQuaternion(robot_orientation)[2]  # Extract yaw angle
        
        # Adjust camera position slightly in front and tilted downward
        camera_eye = np.array(robot_pos) + np.array([0.2, 0, -0.1])  # Move camera forward and raise height
        forward_vector = np.array([np.cos(robot_yaw), np.sin(robot_yaw), -tilt_factor])  # Dynamic tilt
        target_pos = camera_eye + forward_vector

        view_matrix = p.computeViewMatrix(camera_eye.tolist(), target_pos.tolist(), [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near_val, self.far_val)

        img_arr = p.getCameraImage(self.camera_width, self.camera_height, view_matrix, proj_matrix)
        rgb_array = np.reshape(img_arr[2], (self.camera_height, self.camera_width, 4))  # Convert to image format
        
        return rgb_array[:, :, :3]  # Remove alpha channel

    def detect_nearest_object(self):
        """Processes camera feed and detects the nearest object based on color."""
        min_distance = float('inf')
        nearest_cylinder_color = None

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        for cylinder_id, cylinder_color, token in self.cylinders:
            cyl_pos, _ = p.getBasePositionAndOrientation(cylinder_id)
            distance = np.linalg.norm(np.array(robot_pos[:2]) - np.array(cyl_pos[:2]))

            if distance < min_distance:
                min_distance = distance
                nearest_cylinder_color = token  # Use token (color name)

        frame = self.get_camera_feed()  # Use default tilt

        if frame is None or frame.size == 0:
            print("Warning: Empty camera frame!")
            return None, float('inf')

        hsv = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_BGR2HSV)

        # Debugging: Print HSV values at center
        center_x, center_y = self.camera_width // 2, int(self.camera_height * 0.6)
        print(f"HSV Data at ({center_x}, {center_y}): {hsv[center_y, center_x]}")

        # Define color ranges for waste detection
        color_ranges = {
            "Wet Waste": ([100, 150, 50], [140, 255, 255]),  # Blue
            "Dry Waste": ([35, 100, 50], [85, 255, 255]),   # Green
            "Hazardous": ([0, 100, 50], [10, 255, 255]),    # Red
        }

        detected_object = None
        max_area = 0

        for label, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue  # Skip if no object detected

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    detected_object = label

        print(f"Camera Detected Object: {detected_object}")
        print(f"Nearest Cylinder Color: {nearest_cylinder_color}, Distance: {min_distance:.2f}m")

        return detected_object, min_distance


    def move_toward(self):
        """Moves toward detected waste object dynamically, slows down within 0.5m, stops at 0.43m, and tilts camera."""
        if self.attached_object:
            p.resetBaseVelocity(self.robot_id, [0, 0, 0])  # Halt when holding an object
            return

        if self.avoiding_obstacles and time.time() < self.obstacle_avoidance_end_time:
        # Obstacle avoidance mode is active
            p.resetBaseVelocity(self.robot_id, [-self.speed, self.speed, 0])  # Move sideways
            return

        detected_object, distance = self.detect_nearest_object()
    
        if detected_object:
            if distance < 0.48:
                print(f"Stopping: {detected_object} is within {distance:.2f}m")
                self.speed = 0
                tilt_factor = 1.0  # Fully downward when stopped
            elif distance < 0.7:
                self.speed = self.base_speed * ((distance - 0.48) / (0.7 - 0.48))  # Linear slowdown
                tilt_factor = 0.4 + (1.0 - 0.4) * ((0.7 - distance) / (0.7 - 0.48))  # Gradual downward tilt
            else:
                self.speed = self.base_speed  # Full speed when farther than 0.7m
                tilt_factor = 0.4  # Default tilt

            print(f"Moving towards: {detected_object} at {distance:.2f}m away, Speed: {self.speed:.2f}m/s, Tilt Factor: {tilt_factor:.2f}")
            # self.get_camera_feed(tilt_factor)  # Apply the calculated tilt
            p.resetBaseVelocity(self.robot_id, [self.speed, 0, 0])  # Adjust speed dynamically

        else:
            p.resetBaseVelocity(self.robot_id, [0, 0, 0])  # Stop if no object detected


    def update(self):
        """Main update function to process camera input and move accordingly."""
        self.move_toward()  # Move toward an object dynamically