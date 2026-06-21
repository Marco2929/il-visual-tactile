from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'il_main'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'cfg'), glob(os.path.join('cfg', '*.rviz*'))),
    ],
install_requires=['setuptools', 'scipy', 'numpy', 'dex-retargeting'],
    zip_safe=True,
    maintainer='mob-mm',
    maintainer_email='marco.menner@ipa.fraunhofer.de',
    description='UR10e+Artus MoveIt 2 controller consuming predicted actions',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'teleop_ur = il_main.teleop_ur:main',
            'teleop_inspire = il_main.teleop_inspire:main',
            'teleop_inspire_r1 = il_main.teleop_inspire_r1:main',
            'record = il_main.record:main',
            'move_robot_pose = il_main.move_robot_pose:main',
            'control_ur_inspire = il_main.control_ur_inspire:main',
        ],
    },
)

