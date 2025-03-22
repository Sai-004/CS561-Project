import pybullet as p
import numpy as np
import cv2
import time

class Robot:
    def __init__(self, start_pos, cylinders, bins, speed=0.5, camera_enabled=True):
        self.robot_id = p.loadURDF("r2d2.urdf", start_pos)
        self.available_cylinders = cylinders[:]  # Initially, all cylinders are available
        self.collected_cylinders = []  # Stores cylinders that have been picked up and dropped off
        self.bins = {token: pos for _, token, pos in bins}  # Map token to bin position
        self.base_speed = speed
        self.current_speed = 0.0
        self.acceleration = 0.05
        self.camera_enabled = camera_enabled
        self.attached_object = None
        self.target_bin = None  # Bin to move toward after pickup
        self.last_motion_direction = 0  

        # Restrict motion to XY plane and allow only yaw rotation
        p.changeDynamics(self.robot_id, -1, linearDamping=0, angularDamping=0)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        if self.camera_enabled:
            self.setup_camera()

    def setup_camera(self):
        """Initialize camera parameters and position it slightly in front of the robot and tilted downward."""
        self.camera_width = 160
        self.camera_height = 80
        self.fov = 30
        self.aspect = self.camera_width / self.camera_height
        self.near_val = 0.02
        self.far_val = 3.0

    def get_camera_feed(self, tilt_factor=0.4):
        """Captures the robot's camera feed and positions it lower, closer to the feet."""
        robot_pos, robot_orientation = p.getBasePositionAndOrientation(self.robot_id)

        # Get robot velocity
        velocity, _ = p.getBaseVelocity(self.robot_id)
        velocity_x, velocity_y = velocity[:2]
        speed_magnitude = np.linalg.norm([velocity_x, velocity_y])

        if speed_magnitude > 0.2:
            self.last_motion_direction = np.arctan2(velocity_y, velocity_x)

        camera_eye = np.array(robot_pos) + np.array([
            0.2 * np.cos(self.last_motion_direction),
            0.2 * np.sin(self.last_motion_direction),
            -0.1
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

        print("RGB Camera Data:", rgb_array.shape)
        return rgb_array[:, :, :3]

    def detect_nearest_object(self):
        """Finds the nearest available cylinder and returns its distance."""
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
        """Moves toward a detected object, stops at 0.43m, then carries it to the correct bin."""
        robot_pos, robot_orientation = p.getBasePositionAndOrientation(self.robot_id)

        if self.attached_object:
            # Moving to correct bin
            direction = np.array(self.target_bin) - np.array(robot_pos[:2])
            distance = np.linalg.norm(direction)

            if distance < 0.3:  # Stop before dropping
                print(f"Halting before dropping cylinder at {self.target_bin}")
                p.resetBaseVelocity(self.robot_id, [0, 0, 0])
                time.sleep(1)  # Pause before dropping

                print("Dropping cylinder...")
                p.removeConstraint(self.attached_object)  # Release object
                self.collected_cylinders.append(self.attached_object)  # Move to collected list
                self.attached_object = None
                self.current_speed = 0.0  # Reset speed after drop
            else:
                direction = direction / np.linalg.norm(direction)
                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)
                velocity = self.current_speed * direction
                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0], [0, 0, 0])
                print(f"Moving to bin: {self.target_bin}, Speed: {self.current_speed:.2f}")
            return

        nearest_cylinder, distance = self.detect_nearest_object()

        if nearest_cylinder and not self.attached_object:
            cylinder_id, token, cylinder_pos = nearest_cylinder
            direction = np.array(cylinder_pos[:2]) - np.array(robot_pos[:2])
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else [0, 0]

            if distance < 0.45:
                print(f"Halting before attaching to cylinder {cylinder_id}")
                p.resetBaseVelocity(self.robot_id, [0, 0, 0])
                time.sleep(1)  # Pause before attaching

                print(f"Attaching to cylinder {cylinder_id}")
                self.attached_object = p.createConstraint(
                    self.robot_id, -1, cylinder_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, -0.05]
                )

                # Prevent unwanted rotation by locking roll & pitch
                p.changeDynamics(self.robot_id, -1, angularDamping=1.0, mass=10)
                p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

                # Set target bin based on color token
                self.target_bin = self.bins.get(token, (-4, -2.3))  # Default bin if not found
                print(f"Target bin for cylinder {cylinder_id} set to {self.target_bin}")

                # Remove from available list, add to collected list
                self.available_cylinders = [cyl for cyl in self.available_cylinders if cyl[0] != cylinder_id]
                self.current_speed = 0.0  # Reset speed for gradual acceleration
            else:
                self.current_speed = min(self.current_speed + self.acceleration, self.base_speed)
                velocity = self.current_speed * direction
                p.resetBaseVelocity(self.robot_id, [velocity[0], velocity[1], 0], [0, 0, 0])  # Z velocity restricted
                print(f"Moving to cylinder {cylinder_id}, Distance: {distance:.2f}, Speed: {self.current_speed:.2f}")

    def update(self):
        """Main update function to process camera input and move accordingly."""
        self.get_camera_feed()  # Capture and print camera data
        self.move_toward()