# Origin-Submission

ROS 2 solution for TurtleBot3 simulation and trajectory tracking in Gazebo, with an additional node for obstacle avoidance.

The solution lives in the package **`origin`**. All other packages in this workspace come from the official ROBOTIS TurtleBot3 simulation stack and are included only so the project builds and runs out of the box.

## Environment

| Item | Version |
|------|---------|
| OS | Ubuntu 24.04 |
| ROS 2 | Jazzy Jalisco |

## Repository Contents

| Package | Source | Purpose |
|---------|--------|---------|
| `origin` | **This submission** | Trajectory tracking and obstacle avoidance nodes |
| `turtlebot3_*` and related packages | [ROBOTIS TurtleBot3](https://docs.robotis.com/docs/systems/turtlebot3/simulation/gazebo_simulation/) | Robot description, messages, and Gazebo simulation worlds |

## Installation

### 1. Install dependencies

Before building, install the dependencies listed in the official ROBOTIS TurtleBot3 simulation guide:

https://docs.robotis.com/docs/systems/turtlebot3/simulation/gazebo_simulation/

Follow the Jazzy instructions on that page for ROS 2, Gazebo, and the TurtleBot3 simulation requirements.

### 2. Create a workspace and add the packages

```bash
mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src
unzip /path/to/submission.zip -d .
```

Make sure all the included packages (including `origin`) sit directly inside `~/turtlebot3_ws/src`.

### 3. Build

```bash
cd ~/turtlebot3_ws
source /opt/ros/jazzy/setup.bash
colcon build
```

### 4. Source the workspace

```bash
source ~/turtlebot3_ws/install/local_setup.bash
```

## Running the Simulation

You need two terminals. Source the workspace in **each** one.

### Terminal 1: Launch Gazebo

```bash
cd ~/turtlebot3_ws
source install/local_setup.bash
export TURTLEBOT3_MODEL=burger   # see note below
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

Wait until Gazebo has fully loaded with the TurtleBot3 spawned in the world.

### Terminal 2: Run a node

```bash
cd ~/turtlebot3_ws
source install/local_setup.bash
```

Then pick one:

**Trajectory tracking**

```bash
ros2 run origin my_node
```

**Trajectory tracking with obstacle avoidance**

```bash
ros2 run origin my_node_obs
```

## Nodes

| Executable | Description |
|------------|-------------|
| `my_node` | Trajectory tracking in the TurtleBot3 world |
| `my_node_obs` | Trajectory tracking with obstacle avoidance |

## Drive Link
https://drive.google.com/drive/folders/12LeVZR62Zu5OtvGKZfwS50rbZFaooyqC?usp=sharing

## Google Docs Report
https://docs.google.com/document/d/1-SIBh_EA2H4k-rBWkCfzeAbBHax2jBRVDyHH3bqhVaY/edit?usp=sharing
