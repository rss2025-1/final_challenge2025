#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__("traffic_light_detector")
        #NOTE: ConeLocationPixel is a message that already repepesnt a pixel (u,v), but you can just create ur own message called BananaLocationPixel if you want
        self.redlight_pub = self.create_publisher(Bool, "/red_light_detected", 10) 
        self.debug_pub = self.create_publisher(Image, "/red_light_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Traffic Light Detector Initialized")
    def image_callback(self, image_msg):
        """
        Take in image, if red light detected, we want to pubilsh red_light pub to state machine
        """
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")        
        bounding_box = cd_color_segmentation(np.array(image))
        
        red_light_status = Bool()
        red_light_status.data = bounding_box != ((0, 0), (0, 0))            
        self.redlight_pub.publish(red_light_status)
        #For visualization
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = TrafficLightDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
