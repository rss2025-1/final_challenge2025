#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Point
from vs_msgs.msg import ConeLocationPixel
from models.detector import Detector #fix import path, Detector in final_challenge_2025/shrinkrayheist/models
class PersonDetector(Node):
    def __init__(self):
        super().__init__("person_detector")
        self.YOLO = False
        #NOTE: ConeLocationPixel is a message that already repepesnt a pixel (u,v), but you can just create ur own message called PersonLocationPixel if you want
        self.shoe_pub = self.create_publisher(Bool, "/shoe_detected", 10) 
        self.debug_pub = self.create_publisher(Image, "/shoe_debug_img", 10)
        if not self.YOLO:
            self.simple_estop = True
            self.ang_bounds = -np.pi/2, np.pi/2
            self.lidar_dist = 0.1
            self.car_width = 0.2
            callback = self.simple_estop_cb if self.simple_estop_cb else self.complex_estop_cb
     
            self.lidar_sub = self.create_subscription(LaserScan, "/scan", callback, 10)
            
        else:
            self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
            self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
            #todo
        
        self.get_logger().info("person_detector_initialized")

    def image_callback(self, image_msg):
        """
        callback should use YOLO to detect shoe, if shoe detected, stop car
        """ 
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)
    def simple_estop_cb(self, scan_msg):
        """ Processes LIDAR scan data and determines if an emergency stop is needed """
        
       
         
        min_angle_index = len(scan_msg.ranges)//2 - 15
        max_angle_index = len(scan_msg.ranges)//2 + 15
        ranges = np.array(scan_msg.ranges[min_angle_index:max_angle_index+1])
        # self.get_logger().info(f"ranges is {ranges}")
        
        ranges_satisfied = np.sum(ranges < self.estop_dist) 

        should_estop =  ranges_satisfied >= self.count_threshold
        shoe_found = Bool()
        shoe_found.data = should_estop  
        self.shoe_pub.publish(shoe_found)
    def complex_estop_cb(self, scan_msg):
        """ Processes LIDAR scan data and determines if an emergency stop is needed """
    
        angle_start, angle_end = self.ang_bounds
        num_ranges = len(scan_msg.ranges)
        ranges = np.array(scan_msg.ranges)
    
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, num_ranges)
        mask_min_dist = np.where(ranges > self.lidar_dist)

        ranges = ranges[mask_min_dist]
        angles = angles[mask_min_dist]

        scan_polar_vectors = np.vstack((ranges, angles))
        scan_polar_vectors = scan_polar_vectors[:, (scan_polar_vectors[1, :] <= angle_end) & 
                                                (scan_polar_vectors[1, :] >= angle_start)]

        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)

        mask_estop = (np.abs(y_coords) <= self.car_width) & (x_coords <= self.estop_dist)
        close_points_count = np.sum(mask_estop)

        should_estop = (close_points_count >= self.count_threshold)

        shoe_found = Bool()
        shoe_found.data = should_estop  
        self.shoe_pub.publish(shoe_found)
    
def main(args=None):
    rclpy.init(args=args)
    cone_detector = PersonDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
