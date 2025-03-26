import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import random


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
        self.attachment_constraint = None  
        self.target_bin = None  
        self.last_motion_direction = 0  
        self.bin_fill_index = {1: 0, 2: 0, 3: 0}  
        
        # Increase robot mass for better stability
        p.changeDynamics(self.robot_id, -1, mass=100)

        # LiDAR parameters
        self.num_rays = 20  
        self.lidar_range = 1.5  

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

    def setup_camera(self):
        self.camera_width = 160
        self.camera_height = 80
        self.fov = 30
        self.aspect = self.camera_width / self.camera_height
        self.near_val = 0.02
        self.far_val = 3.0

    def detect_obstacles_lidar(self, min_detection_radius=0.15):
        """Use LiDAR to detect obstacles outside a given radius and return avoidance direction"""
        robot_pos, robot_orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_yaw = p.getEulerFromQuaternion(robot_orientation)[2]

        angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_rays)  
        ray_from = []
        ray_to = []

        for angle in angles:
            ray_direction = np.array([np.cos(robot_yaw + angle), np.sin(robot_yaw + angle), 0])
            start_pos = np.array(robot_pos) + np.array([0, 0, 0.2])  
            end_pos = start_pos + self.lidar_range * ray_direction
            ray_from.append(start_pos)
            ray_to.append(end_pos)

        results = p.rayTestBatch(ray_from, ray_to)

        min_distance = self.lidar_range
        best_angle = 0
        obstacle_detected = False

        for i, result in enumerate(results):
            hit_fraction = result[2]  
            if hit_fraction < 1.0:  # Obstacle detected
                distance = hit_fraction * self.lidar_range

                if distance > min_detection_radius:  # Ignore close obstacles
                    obstacle_detected = True
                    if distance < min_distance:
                        min_distance = distance
                        best_angle = angles[i]

        if obstacle_detected:
            print(f"Obstacle detected at {min_distance:.2f}m. Avoiding...")

            return 0.5 if best_angle < 0 else -0.5  

        return 0  

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

    def move_toward(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)

        if self.attached_object:
            direction = np.array(self.target_bin[:2]) - np.array(robot_pos[:2])
            distance = np.linalg.norm(direction)

            if distance < 1.8:
                p.resetBaseVelocity(self.robot_id, [0, 0, 0])

                bin_size = 2.0  
                grid_size = 3  
                cell_spacing = bin_size / (grid_size + 1)  

                bx, by = self.target_bin  
                row = self.bin_fill_index[self.target_bin_token] // grid_size
                col = self.bin_fill_index[self.target_bin_token] % grid_size

                drop_x = bx - bin_size / 2 + (col + 1) * cell_spacing
                drop_y = by - bin_size / 2 + (row + 1) * cell_spacing
                drop_position = (drop_x, drop_y)

                self.bin_fill_index[self.target_bin_token] += 1  

                if self.attachment_constraint is not None:
                    p.removeConstraint(self.attachment_constraint)
                    self.attachment_constraint = None  

                p.resetBasePositionAndOrientation(self.attached_object, 
                                                  [drop_position[0], drop_position[1], 1],  
                                                  [0, 0, 0, 1])

                self.collected_cylinders.append(self.attached_object)
                self.attached_object = None  
                self.current_speed = 0.0  

            else:
                avoid_direction = self.detect_obstacles_lidar()
                direction = direction / np.linalg.norm(direction)

                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)

                velocity = self.current_speed * direction
                if avoid_direction != 0:
                    velocity = [velocity[1] * avoid_direction, -velocity[0] * avoid_direction, 0]

                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0])
            return

        nearest_cylinder, distance = self.detect_nearest_object()

        if nearest_cylinder and not self.attached_object:
            cylinder_id, token, cylinder_pos = nearest_cylinder

            if token == 4:
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                return

            direction = np.array(cylinder_pos[:2]) - np.array(robot_pos[:2])
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else [0, 0]

            if distance < 0.48:
                p.resetBaseVelocity(self.robot_id, [0, 0, 0])

                self.attachment_constraint = p.createConstraint(
                    self.robot_id, -1, cylinder_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, -0.05]
                )
                self.attached_object = cylinder_id  
                self.target_bin = self.bins.get(token, (-4, -2.3))  
                self.target_bin_token = token  

                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                self.current_speed = 0.0  
            else:
                avoid_direction = self.detect_obstacles_lidar()
                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)
                velocity = self.current_speed * direction

                if avoid_direction != 0:
                    velocity = [velocity[1] * avoid_direction, -velocity[0] * avoid_direction, 0]

                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0])

    def update(self):
        self.get_camera_feed()
        self.move_toward()
