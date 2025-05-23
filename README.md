# Anonymous Trash Collection Robot

This project implements an autonomous trash collection robot using PyBullet for simulation. The robot navigates a virtual environment to collect cylinders (representing trash) and sorts them into appropriate bins. The project includes two implementations:

1. **CNN and RL Implementation**: Combines a Convolutional Neural Network (CNN) for object classification with Reinforcement Learning (RL) (Q-Learning) for decision-making.
2. **Object Detection in PyBullet Implementation**: Uses depth-based object detection.

The project is organized into two folders, each containing a complete implementation with its own simulation environment.

## Project Structure

- **cnn_and_rl/**: Contains the CNN and RL implementation.
  - `world.py`: Main script to run the simulation.
  - `robot.py`: Defines the `Robot` class with CNN-based classification and RL logic.
  - Other supporting files (e.g., trained CNN model, URDF models).
- **object_detection_in_pybullet/**: Contains the object detection implementation.
  - `world.py`: Main script to run the simulation.
  - `robot.py`: Defines the `Robot` class with depth-based detection.
  - Other supporting files (e.g., trained CNN model, URDF models).
- **requirements.txt**: Lists the Python dependencies required for both implementations.

## Prerequisites

- Python 3.8 or higher
- Microsoft Visual C++ Build Tools (download from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/))
- A virtual environment (recommended)
- Windows, macOS, or Linux

## Installation

1. **Clone the Repository** (or download the project files):
   ```bash
   git clone https://github.com/Sai-004/CS561-Project
   cd CS561-Project
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv env
   ```
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
     If the above fails due to absence of the required Execution Policy, the following command would temporarily enable the script to run:
     ```bash
     Set-ExecutionPolicy -Scope CurrentUser Unrestricted
     ```
     Use the above, then retry the previous command to run the script.
     Also, due to Windows restriction on path length, try running the code in a shallow directory, i.e. whose absolute path length is lesser.
     Even after this, there may be some system dependent modifications required, which would be as per your system. Please consult online help for doing so.
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs the required packages: `pybullet`, `numpy`, `opencv-python`, and `tensorflow`.

## Running the Implementations

Each implementation can be run by navigating to its respective folder and executing the `world.py` script. Ensure the virtual environment is activated before running the commands.

### 1. CNN and RL Implementation
This implementation uses a CNN to classify trash (red, green, blue cylinders) and RL to decide where to place them in a tray.

**Steps**:
```bash
cd cnn_and_rl
python world.py
```
![image](https://github.com/user-attachments/assets/48f09d23-8649-4221-ba81-b216e9f5aa8e)
*The robot with bins covered by transparent cover.*

![image](https://github.com/user-attachments/assets/0e619325-d5a0-4aba-bdc4-cfa7361b7ab3)
*The robot stopped at a safe distance from the object, took screenshot
and will proceed with cnn classification and q-learning.*

**What to Expect**:
- A PyBullet simulation window opens, showing an R2D2-like robot.
- The robot navigates to detect and collect cylinders, classifying them using the CNN.
- Cylinders are sorted into a tray behind the robot based on RL decisions.
- Console output shows classification results, RL actions, and simulation status.

### 2. Object Detection in PyBullet Implementation
This implementation uses depth-based detection to locate trash, ignoring the brown floor.

**Steps**:
```bash
cd object_detection_in_pybullet
python world.py
```

**What to Expect**:
- A PyBullet simulation window opens, showing the same R2D2-like robot.
- The robot uses depth images to detect cylinders, filters out the floor.
- Console output logs detection details.

## Requirements
The `requirements.txt` file includes:
```
opencv-python==4.11.0.86
pybullet==3.2.7
tensorflow==2.19.0
```
These packages cover all dependencies needed for both implementations. `numpy` is installed automatically as a dependency of `opencv-python` and `tensorflow`.

## Notes
- **Environment**: Use a virtual environment to avoid conflicts with other Python projects.
- **Simulation**: The PyBullet GUI may require a display. If running on a server, consider using a virtual display or headless mode.
- **Model Files**: Ensure the `GarbageCollectorCNN.h5` model file is present in both folders, as it’s used for cylinder classification.
- **Troubleshooting**: If you encounter version conflicts, try updating `requirements.txt` with compatible versions or installing packages without version pins (e.g., `pip install opencv-python`).

## License
This project is for educational purposes and part of a CS561 AI course assignment.
