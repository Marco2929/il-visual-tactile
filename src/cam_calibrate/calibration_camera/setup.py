import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'calibration_camera'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='you@email.com',
    description='Hand-Eye Calibration for UR10e with ZED 2i and RealSense 435i',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'calibration_node = calibration_camera.calibration_node:main',
            'tcp_pose_publisher = calibration_camera.tcp_pose_publisher:main',
            'rgbd_service_zed = calibration_camera.rgbd_service_zed:main',
            'rgbd_service_realsense = calibration_camera.rgbd_service_realsense:main',
        ],
    },
)