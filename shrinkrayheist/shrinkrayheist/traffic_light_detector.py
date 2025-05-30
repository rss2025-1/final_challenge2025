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
from shrinkrayheist.computer_vision.color_segmentation import cd_color_segmentation

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
        h, w = image.shape[:2]

        # Compute bounds for the middle half
        h_start = h // 4 
        h_end = 2 * h // 3

        image = image[h_start:h_end, :]        
        bounding_box = cd_color_segmentation(np.array(image))
        # self.get_logger().info(f"{bounding_box}")
        red_light_status = Bool()
        red_light_status.data = (bounding_box != ((0, 0), (0, 0)))            
        self.redlight_pub.publish(red_light_status)
        #For visualization
        if red_light_status.data:
            top_left, bottom_right = bounding_box
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Red box, thickness 2
            
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")

        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = TrafficLightDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
