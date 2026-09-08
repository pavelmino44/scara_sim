## Dependencies
List of dependencies for ROS2 Humble:
- ros-humble-xacro
- ros-humble-ros-gz-sim
- ros-humble-ros-gz-bridge
- ros-humble-ros2-control
- ros-humble-ros2-controllers
- ros-humble-ign-ros2-control

2. Install dependencies
```bash
sudo apt update
sudo apt install \
    ros-humble-xacro \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ign-ros2-control
```

## Install and Start

1. Download pkg
```bash
mkdir -p some_ws/src
cd some_ws/src
git clone https://github.com/cyberbanana777/scara_sim
```

2. Build 
```bash
cd ..
colcon build --symlink-install && source install/local_setup.bash
```

3. Launch simulation
```bash
ros2 launch scara_sim scara_sim.launch.py 
```

4. Run effort commands. In the new terminal:
```bash
cd some_ws
source install/local_setup.bash
ros2 launch scara_sim effort_publisher.launch.py
```
