from setuptools import find_packages, setup

package_name = 'heist'

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
    maintainer_email='skrem@mit.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ "state_machine = heist.state_machine:main",
                             "traffic_light_detector = heist.traffic_light_detector:main",
                             "person_detector = heist.person_detector:main"
                             "banana_detector = heist.banana_detector:main"
        ],
    },
)
