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
from models.detector import Detector #this is YOLO
from nav_msgs.msg import Odometry

class BananaDetector(Node):
    def __init__(self):
        super().__init__("banana_detector")
        #NOTE: 1. ConeLocationPixel is a message that already repepesnt a pixel (u,v), but you can just create ur own message called BananaLocationPixel if you want
        #NOTE: 2. Create yaml file for all the topics so we don't name something wrong by accident
        self.banana_pub = self.create_publisher(ConeLocationPixel, "/relative_banana_px", 10) 
        self.banana_state = self.create_publisher(Bool, "/banana_detected", 10)  
        self.odom_sub = self.create_subscription(Odometry,"/pf/pose/odom", self.pose_callback, 10)
        self.debug_pub = self.create_publisher(Image, "/banana_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.get_logger().info("Banana Detector Initialized")
    
    def image_callback(self, image_msg):
        """
        callback should use YOLO to detect banana, if banana detected reroute the path
        """ 
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)
    def banana_collected(self):
        """
        find a way to change banana state to false once we have driven over the banana pixel coordinate
        """
        pass
def main(args=None):
    rclpy.init(args=args)
    cone_detector = BananaDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
