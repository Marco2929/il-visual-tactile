<p align="right">
<img src="/dg_description/image/title.svg"/>
</p>
<p align="center">
  <img src="https://tesollo.com/wp-content/uploads/2025/06/DG-5F-6-1.webp" alt="dg5f" width="250px"/>
  <img src="https://tesollo.com/wp-content/uploads/2025/06/DG-4F-1-1.webp" height="180px"/>
  <img src="https://tesollo.com/wp-content/uploads/2024/12/DG-3F-5-1.webp" width="150px"/>
  <img src="https://tesollo.com/wp-content/uploads/2025/02/3F-CES.gif" width="150px"/>
</p>


The DELTO_M_ROS2 Repository is a comprehensive ROS 2 package designed to support the Delto Gripper-M. This project includes simulations, control interfaces, and visualization tools for the gripper, enabling developers to efficiently develop and test robotic applications.


## 📌 **Supported ROS Distributions**

|  **ROS Version** |  **Ubuntu Version** |  **Branch** | **Build Status** |
|------------------|----------------------|---------------|---------------|
| ROS 2            | 22.04 (Jammy)        | `humble`        | ![Build Status](https://github.com/tesollodelto/delto_m_ros2/actions/workflows/humble-ci.yaml/badge.svg?branch=humble) ![Build Status](https://github.com/tesollodelto/delto_m_ros2/actions/workflows/jazzy-ci.yaml/badge.svg?branch=jazzy-dev) |


---

## 📦 Package Structure


```text
DELTO_M_ROS2/
├─ dg3f_m_driver/        # ros2_control driver/interface for the 3-Finger Gripper
├─ dg3f_m_gz/            # 3F Ignition Gazebo simulation + ros2_control + launch
├─ dg4f_driver/          # ros2_control driver/interface for the 4-Finger Gripper
├─ dg4f_gz/              # 4F Ignition Gazebo simulation + ros2_control + launch
├─ dg5f_driver/          # ros2_control driver/interface for the 5-Finger Gripper
├─ dg5f_gz/              # 5F Ignition Gazebo simulation + ros2_control + launch
├─ dg_description/       # Shared URDF/Xacro, meshes, and RViz configs (3F/4F/5F)
├─ dg_msgs/              # Custom ROS 2 message/service/action definitions
├─ dg_sdk_ros2_bridge/   # Bridge between TESOLLO SDK and ROS 2 (drivers, demos)
└─ dg_isaacsim/          # Isaac Sim integration demo (ROS 2 bridge, sample scenes/launch)
```



| Package                                       | Role                                                             | Typical Use                    |
| --------------------------------------------- | ---------------------------------------------------------------- | ------------------------------ |
| [`dg3f_m_driver`](./dg3f_m_driver/)           | Hardware driver for the 3F gripper using `ros2_control`.         | Real hardware control (3F)     |
| [`dg3f_m_gz`](./dg3f_m_gz/)                   | Ignition Gazebo package for 3F, including URDF and launch files. | Simulation & testing           |
| [`dg4f_driver`](./dg4f_driver/)               | Hardware driver for the 4F gripper using `ros2_control`.         | Real hardware control (4F)     |
| [`dg4f_gz`](./dg4f_gz/)                       | Ignition Gazebo package for 4F.                                  | Simulation & testing           |
| [`dg5f_driver`](./dg5f_driver/)               | Hardware driver for the 5F gripper using `ros2_control`.         | Real hardware control (5F)     |
| [`dg5f_gz`](./dg5f_gz/)                       | Ignition Gazebo package for 5F.                                  | Simulation & testing           |
| [`dg_description`](./dg_description/)         | Common URDF/Xacro models, meshes, and RViz configs.              | Shared by all grippers         |
| [`dg_msgs`](./dg_msgs/)                       | Custom message, service, and action definitions.                 | Shared across drivers & bridge |
| [`dg_sdk_ros2_bridge`](./dg_sdk_ros2_bridge/) | TESOLLO SDK ↔ ROS 2 bridge node and utilities.                   | SDK-based communication        |
| [`dg_isaacsim`](./dg_isaacsim/)               | Isaac Sim integration demo with ROS 2 bridge and launch files.   | Full simulation workflow       |


> **Usage Tips**
>
> * Use `*_driver` packages for **real hardware control**.
> * Use `*_gz` packages for **Ignition Gazebo simulation**.
> * `dg_msgs` centralizes custom interfaces, enabling reusability across all drivers and bridges.
> * `dg_description` keeps URDFs and meshes consistent across hardware and simulation.

---

## 🛠️ Installation and Build Instructions

To install and build the DELTO_M_ROS2 project, follow these steps:

1. **Create Workspace and Clone Source Code**

   ```bash
   mkdir -p ~/your_ws/src
   cd ~/your_ws/src
   git clone <DELTO_M_ROS2_repository_URL>
   ```

2. **Install Dependencies**

   ```bash
   cd ~/delto_m_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. **Build Packages**

   ```bash
   colcon build
   ```

4. **Source the Environment**

   ```bash
   source install/setup.bash
   ```


## 🎯 **Performance Demonstrations**

### **Delto Gripper-3F: Advanced Packaging Solutions**

[![Delto Gripper-3F Advanced Packaging Solutions](https://img.youtube.com/vi/x6QCdgl5r8Q/sddefault.jpg)](https://www.youtube.com/watch?v=x6QCdgl5r8Q)  
▶️ *Click the image to watch the video*


### **Delto Gripper-5F: Paper Cup Removal Demonstration**

[![Delto Gripper-5F Paper Cup Removal](https://img.youtube.com/vi/MlUSlto5R9U/sddefault.jpg)](https://www.youtube.com/watch?v=MlUSlto5R9U)  
▶️ *Click the image to watch the video*

## 🤝 Contributing

The DELTO_M_ROS2 project is open-source, and contributions are welcome. To contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Add my feature'`).
4. Push to your branch (`git push origin feature/my-feature`).
5. Open a pull request detailing your modifications.


## 📄 License

This project is released under the BSD-3-Clause license.


## 📧 Contact

For additional support or inquiries about this project, please contact [TESOLLO SUPPORT](mailto:support@tesollo.com). 
