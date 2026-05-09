import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'cobot2'
resource_files = [
    path for path in glob(os.path.join('resource', '*'))
    if os.path.isfile(path) and os.path.basename(path) != package_name
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), resource_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kim',
    maintainer_email='jongun1203@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'object_detection_node = cobot2.detection:main',
            'object_detection_node2 = cobot2.detection2:main',
            'object_detection_service = cobot2.detection3_service:main',
            'detect_cal_pos_service = cobot2.detect_cal_pos_service:main',
            'cal_position = cobot2.cal_position:main',
            'realsense_service = cobot2.realsense2:main',
            'robot_move4 = cobot2.robot_move4:main',
            'robot_move_total = cobot2.robot_move_total:main',
            'firebase_bridge_node = cobot2.firebase_bridge_node:main',
        ],
    },
)
