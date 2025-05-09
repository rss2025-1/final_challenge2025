from setuptools import find_packages, setup
import glob
import os

package_name = 'shrinkrayheist'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/shrinkrayheist/launch', glob.glob(os.path.join('launch', '*launch.*')))

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='skrem@mit.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ "state_machine = shrinkrayheist.state_machine:main",
                             "traffic_light_detector = shrinkrayheist.traffic_light_detector:main",
                             "person_detector = shrinkrayheist.person_detector:main",
                             "banana_detector = shrinkrayheist.banana_detector:main",
                             "autonomous_replanner = shrinkrayheist.autonomous_replanner:main",
                             "u_turn = shrinkrayheist.u_turn:main",
        ],
    },
)
