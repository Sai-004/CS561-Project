# CS561-Project

# Detailed Explanation of the Workspace

This document explains every function and important line of code in the workspace.

## world.py: Simulation Environment Setup

### Imports and PyBullet Setup
- The file imports required libraries such as PyBullet, pybullet_data, time, random, cv2, and numpy, plus the Robot class from `robot.py`.
- **Connection & Configuration:**
  - `p.connect(p.GUI)` launches the PyBullet GUI.
  - `p.setAdditionalSearchPath(pybullet_data.getDataPath())` ensures PyBullet can find required URDF files.
  - `p.setGravity(0, 0, -15.8)` sets the gravity for the simulation.

### Creating the Ground Plane
- `p.loadURDF("plane.urdf")` loads the ground plane onto which all objects (walls, bins, cylinders, robot) will interact.

### Defining Arena Dimensions and Parameters
- Variables such as `arena_length`, `arena_width`, `wall_thickness`, `wall_height`, and `safe_zone_radius` define the sizes for the simulated arena and the safe spawn zone for the robot.

### Creating Walls
- **Wall Positions & Sizes:**
  - `wall_positions` defines four positions around the arena.
  - `wall_sizes` defines each wall’s dimensions.
- **Instantiating Fixed Walls:**
  - For each wall, a collision shape is created using `p.createCollisionShape`.
  - A static wall body (mass = 0) is created with `p.createMultiBody`, ensuring that the wall remains immovable.
  - High lateral friction is set using `p.changeDynamics` to avoid unintended sliding.
  - All created walls are stored in the list `walls`.

### Setting Up Bins
- **Color Tokens and Positions:**
  - The dictionary `COLOR_TOKENS` maps waste types to tokens.
  - Three bin positions are defined, and colors are assigned (blue, green, red) via their RGBA values.
- **Creating Bins:**
  - For each bin, a collision shape is created using `p.createCollisionShape` (with an increased size).
  - A static multi-body is created at the given position.
  - Visual appearance is changed by calling `p.changeVisualShape`.
  - Each bin is stored as a tuple `(bin body, token, position)` in the list `bins`, and their positions are also saved in `bin_areas` to help avoid cylinder overlap.

### Creating Cylinders
- **Cylinder Colors and Tokens:**
  - A mapping (`cylinder_colors`) provides RGBA colors for different waste types.
- **Spawn Position Check:**
  - The helper function `is_valid_spawn(x, y, radius)` ensures that a new cylinder is not too close to an already placed cylinder or any bin.
- **Spawning Loop:**
  - For each waste type, 10 cylinders are spawned.
  - Random positions are generated within the arena (but outside the safe zone) until a valid position is found.
  - A collision shape is created via `p.createCollisionShape`, and a cylinder body is instantiated with mass (10).
  - Its visual appearance is set using `p.changeVisualShape`.
  - Each cylinder is added as a tuple `(cylinder ID, color token, cylinder number)` to the list `cylinders`.

### Instantiating and Running the Robot
- A new `Robot` object is created with its starting position, the list of cylinders, bins, and a specified speed. This invokes the constructor defined in `robot.py`.
- **Simulation Loop:**
  - The simulation runs for 5000 iterations.
  - In each iteration, the robot’s `update()` method is called to handle movement, attachment, and rebalancing.
  - The simulation state updates with `p.stepSimulation()`.
  - If the camera is enabled, the latest camera feed is fetched and displayed using OpenCV (`cv2.imshow` and `cv2.waitKey`).
  - A brief sleep controls the simulation speed.
- **Cleanup:**
  - After the loop, all OpenCV windows are closed with `cv2.destroyAllWindows()`.
  - The simulation disconnects via `p.disconnect()`.

## robot.py: Robot Behavior and Control

### Constructor (`__init__`)
- **Loading the Robot:**
  - The robot is loaded with its URDF via `p.loadURDF("r2d2.urdf", start_pos)`.
- **Cylinders & Bins Setup:**
  - A copy of the provided cylinders is made (`self.available_cylinders`) to later decide which objects to pick.
  - An empty list `self.collected_cylinders` stores picked cylinders.
  - A dictionary (`self.bins`) maps bin tokens to their positions for quickly determining a drop-off location.
- **Speed and Motion Attributes:**
  - Base speed (`base_speed`), current speed, and acceleration are initialized.
  - `self.last_motion_direction` keeps track of the last moving direction for camera alignment.
- **Attachment Attributes:**
  - `self.attached_object` holds the current object being carried.
  - `self.attachment_constraint` will store the constraint ID if a cylinder is attached to the robot.
- **Grid Placement for Bins:**
  - `self.bin_fill_index` helps determine placement positions inside bins using a grid layout.
- **Dynamics Tweaks:**
  - The robot’s mass is increased (`p.changeDynamics`) for better stability.
  - Linear and angular damping are set to zero.
  - The robot’s base velocity is reset.
- **Bin Corners:**
  - The robot calculates the bin corners based on a fixed bin size to eventually place cylinders on a grid inside the bins.
  - The bin corner positions are printed for debugging.
- **Camera Setup:**
  - If `camera_enabled` is true, `setup_camera()` is called to initialize camera parameters.

### Method: `setup_camera()`
- **Purpose:** Sets up camera parameters needed to compute the view.
- **How It Works:** Stores these parameters in instance variables.

### Method: `get_camera_feed(tilt_factor=0.4)`
- **Purpose:** Computes and returns a camera view of the simulation based on the robot’s position and movement.

### Method: `detect_nearest_object()`
- **Purpose:** Scans through all available cylinders and returns the nearest one along with its distance.

### Method: `move_toward()`
- **Purpose:** Directs the robot toward a target, either moving to pick up a cylinder or delivering a cylinder to a bin.

### Method: `check_and_rebalance()`
- **Purpose:** Ensures that the robot does not tip over.

### Method: `update()`
- **Purpose:** The main update loop method, called every simulation step.

## Summary

### world.py:
- Sets up the simulation with a ground plane, walls, bins, and cylinders.
- Instantiates the robot and drives a simulation loop that calls the robot’s `update()` method for each step.
- Manages environment configuration and cleanup.

### robot.py:
- Defines a `Robot` class that:
  - Loads a robot URDF and configures its dynamics.
  - Sets up a camera to view the simulation.
  - Continuously scans for and picks up the nearest cylinder (ignoring “Not Waste”).
  - Moves toward designated bins, drops cylinders using a grid layout, and rebalances itself when tilted.
- The `update()` method ties together periodic stability checks, camera updates, and movement logic.


