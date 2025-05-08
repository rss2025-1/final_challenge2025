from setuptools import find_packages, setup

package_name = 'lane_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='shrika@mit.edu',
    description='Lane follower implementation for the Race to the Moon challenge',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follower = lane_follower.lane_follower:main',
            'yessir = lane_follower.yessir:main'
        ],
    },
)
