from setuptools import setup

package_name = 'vision_yolov7'
models = "vision_yolov7/models"
utils = "vision_yolov7/utils"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robofei',
    maintainer_email='luanawt43@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect = vision_yolov7.detect:main',
            'posicaomotor = vision_yolov7.posicaomotor:main'
        ],
    },
)
