#NOTE: if we are doing YOLO for shoes, then do this code, otherwise upgrade safety controller and put in launch file

#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from vs_msgs.msg import ConeLocationPixel
from models.detector import Detector #fix import path, Detector in final_challenge_2025/shrinkrayheist/models
class PersonDetector(Node):
    def __init__(self):
        super().__init__("person_detector")
        #NOTE: ConeLocationPixel is a message that already repepesnt a pixel (u,v), but you can just create ur own message called PersonLocationPixel if you want
        self.shoe_pub = self.create_publisher(Bool, "/shoe_detected", 10) 
        self.debug_pub = self.create_publisher(Image, "/shoe_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.get_logger().info("person_detector_initialized")
    def image_callback(self, image_msg):
        """
        callback should use YOLO to detect shoe, if shoe detected, stop car
        """ 
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)
   
def main(args=None):
    rclpy.init(args=args)
    cone_detector = PersonDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
