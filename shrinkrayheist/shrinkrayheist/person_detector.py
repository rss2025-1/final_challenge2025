#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Point
# from model.detector import Detector #fix import path, Detector in final_challenge_2025/shrinkrayheist/models
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
            self.lidar_dist = 0.05
            self.car_width = 0.2
            self.estop_dist = 0.5
            self.count_threshold = 12 #6
            callback = self.simple_estop_cb if self.simple_estop_cb else self.complex_estop_cb
            self.prev_valid_count = 0
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
        min_angle_index = len(scan_msg.ranges)//2 - 23
        max_angle_index = len(scan_msg.ranges)//2 + 23
        ranges = np.array(scan_msg.ranges[min_angle_index:max_angle_index+1])
        
        ranges_satisfied = np.sum(ranges < self.estop_dist)

        # if ranges_satisfied == 6:
        #     self.get_logger().warn(f"Detected exactly 6 points, using previous valid count: {self.prev_valid_count}")
        #     ranges_satisfied = self.prev_valid_count
        # else:
        #     self.prev_valid_count = ranges_satisfied  # Update last valid value

        # self.get_logger().info(f"ranges_satisfied = {ranges_satisfied}, threshold = {self.count_threshold}")

        should_estop = bool(ranges_satisfied >= self.count_threshold)

        shoe_found = Bool()
        shoe_found.data = should_estop  
        self.shoe_pub.publish(shoe_found)
    def complex_estop_cb(self, scan_msg):
        """ Processes LIDAR scan data and determines if an emergency stop is needed """

        angle_start, angle_end = self.ang_bounds
        num_ranges = len(scan_msg.ranges)
        
        #mask_min_dist = np.where(scan_msg.ranges > self.lidar_dist)
        #ranges = np.array(scan_msg.ranges)[mask_min_dist]
        angles = np.linspace(scan_msg.angle_min,scan_msg.angle_max, num_ranges)
        angle_mask = (angles >= angle_start) and (angles <= angle_end)
        ranges = ranges[angle_mask]
        angles = angles[angle_mask]

        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)

        mask_estop = (np.abs(y_coords) <= self.car_width) and (x_coords <= self.estop_dist)
        close_points_count = np.sum(mask_estop)

        should_estop = bool(close_points_count >= self.count_threshold)

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
# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# import numpy as np
# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import Bool
# from sensor_msgs.msg import Image, LaserScan
# from geometry_msgs.msg import Point
# from shrinkrayheist.model.detector import Detector # yolo nodel

# class PersonDetector(Node):
#     def __init__(self):
#         super().__init__("person_detector")

#         self.use_yolo = False
#         self.use_lidar = True

#         # shared publisher for stop signal
#         # self.estop_pub = self.create_publisher(Bool, "/shoe_detected", 10)
#         self.shoe_pub = self.create_publisher(Bool, "/shoe_detected", 10) 
#         # YOLO
#         if self.use_yolo:
#             self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.yolo_cb, 5)
#             self.bridge = CvBridge()
#             self.debug_pub = self.create_publisher(Image, "/shoe_debug_img", 10)
#             self.detector = Detector()
#             self.detector.set_threshold(0.5)
#             self.yolo_stop_active = False
#             self.yolo_classes = {"person", "shoe", "foot", "feet"}

#         # LIDAR
#         if self.use_lidar:
#             self.simple_estop = True
#             self.ang_bounds = -np.pi/2, np.pi/2
#             self.lidar_dist = 0.1
#             self.car_width = 0.2
#             self.estop_dist = 1.0
#             self.count_threshold = 6

#             if self.simple_estop:
#                 self.lidar_sub = self.create_subscription(LaserScan, "/scan", self.simple_estop_cb, 10)
#             else:
#                 self.lidar_sub = self.create_subscription(LaserScan, "/scan", self.complex_estop_cb, 10)

#         self.get_logger().info("person_detector_initialized")

#     def publish_estop(self, should_stop: bool):
#         msg = Bool()
#         msg.data = bool(should_stop)
        
#         self.estop_pub.publish(msg)

#     def yolo_cb(self, image_msg):
#         """Uses YOLO to detect shoes or people, and publishes stop if needed."""
#         try:
#             image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
#         except CvBridgeError as e:
#             self.get_logger().error(f"cv_bridge error: {e}")
#             return

#         person_detected = False
#         results = self.detector.predict(image)
#         predictions = results["predictions"]
#         debug_image = results["original_image"]

#         for (x1, y1, x2, y2), label in predictions:
#             if label.lower() in self.yolo_classes:
#                 person_detected = True

#         if person_detected:
#             self.get_logger().info("YOLO: Person detected. Triggering estop.")
#         self.publish_estop(person_detected)

#         # publish debug image
#         pil_debug_image = self.detector.draw_box(debug_image, predictions, draw_all=True)
#         debug_image = np.array(pil_debug_image)
#         debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
#         debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
#         self.debug_pub.publish(debug_msg)

#     def simple_estop_cb(self, scan_msg):
#         """LIDAR estop logic - simple forward-facing band."""
#         mid_idx = len(scan_msg.ranges) // 2
#         min_angle_index = mid_idx - 15
#         max_angle_index = mid_idx + 15

#         ranges = np.array(scan_msg.ranges[min_angle_index:max_angle_index+1])
#         close_count = np.sum(ranges < self.estop_dist)

#         should_stop = bool(close_count >= self.count_threshold)
#         # if should_stop:
#             # self.get_logger().info("LIDAR: Obstacle detected. Triggering estop.")
#         # self.publish_estop(should_stop)
#         shoe_found = Bool()
#         shoe_found.data = should_stop  
#         self.shoe_pub.publish(shoe_found)

#     def complex_estop_cb(self, scan_msg):
#         """Advanced LIDAR estop using geometry filtering."""
#         angle_start, angle_end = self.ang_bounds
#         ranges = np.array(scan_msg.ranges)
#         angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

#         valid = bool(ranges > self.lidar_dist)
#         ranges, angles = ranges[valid], angles[valid]

#         mask = (angles >= angle_start) & (angles <= angle_end)
#         ranges, angles = ranges[mask], angles[mask]

#         x = ranges * np.cos(angles)
#         y = ranges * np.sin(angles)

#         close_mask = (np.abs(y) <= self.car_width) & (x <= self.estop_dist)
#         should_stop = np.sum(close_mask) >= self.count_threshold

#         if should_stop:
#             self.get_logger().info("LIDAR: Complex estop triggered.")
#         self.publish_estop(should_stop)

# def main(args=None):
#     rclpy.init(args=args)
#     detector_node = PersonDetector()
#     rclpy.spin(detector_node)
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()




# # #!/usr/bin/env python
# # import rclpy
# # from rclpy.node import Node
# # import numpy as np
# # import cv2
# # from cv_bridge import CvBridge, CvBridgeError
# # from std_msgs.msg import Bool
# # from sensor_msgs.msg import Image, LaserScan
# # from geometry_msgs.msg import Point
# # # from model.detector import Detector #fix import path, Detector in final_challenge_2025/shrinkrayheist/models
# # class PersonDetector(Node):
# #     def __init__(self):
# #         super().__init__("person_detector")
# #         self.YOLO = False
# #         #NOTE: ConeLocationPixel is a message that already repepesnt a pixel (u,v), but you can just create ur own message called PersonLocationPixel if you want
# #         self.shoe_pub = self.create_publisher(Bool, "/shoe_detected", 10) 
# #         self.debug_pub = self.create_publisher(Image, "/shoe_debug_img", 10)
# #         if not self.YOLO:
# #             self.simple_estop = True
# #             self.ang_bounds = -np.pi/2, np.pi/2
# #             self.lidar_dist = 0.1
# #             self.car_width = 0.2
# #             self.estop_dist = 1.0
# #             self.count_threshold = 6
# #             callback = self.simple_estop_cb if self.simple_estop_cb else self.complex_estop_cb
     
# #             self.lidar_sub = self.create_subscription(LaserScan, "/scan", callback, 10)
            
# #         else:
# #             self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
# #             self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
# #             #todo
        
# #         self.get_logger().info("person_detector_initialized")

# #     def image_callback(self, image_msg):
# #         """
# #         callback should use YOLO to detect shoe, if shoe detected, stop car
# #         """ 
# #         image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
# #         debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
# #         self.debug_pub.publish(debug_msg)
# #     def simple_estop_cb(self, scan_msg):
# #         """ Processes LIDAR scan data and determines if an emergency stop is needed """
# #         min_angle_index = len(scan_msg.ranges)//2 - 15
# #         max_angle_index = len(scan_msg.ranges)//2 + 15
# #         ranges = np.array(scan_msg.ranges[min_angle_index:max_angle_index+1])
# #         # self.get_logger().info(f"ranges is {ranges}")
        
# #         ranges_satisfied = np.sum(ranges < self.estop_dist) 

# #         should_estop =  bool(ranges_satisfied >= self.count_threshold)
# #         shoe_found = Bool()
# #         shoe_found.data = should_estop  
# #         self.shoe_pub.publish(shoe_found)
# #     def complex_estop_cb(self, scan_msg):
# #         """ Processes LIDAR scan data and determines if an emergency stop is needed """
    
# #         angle_start, angle_end = self.ang_bounds
# #         num_ranges = len(scan_msg.ranges)
# #         ranges = np.array(scan_msg.ranges)
    
# #         angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, num_ranges)
# #         mask_min_dist = np.where(ranges > self.lidar_dist)

# #         ranges = ranges[mask_min_dist]
# #         angles = angles[mask_min_dist]

# #         scan_polar_vectors = np.vstack((ranges, angles))
# #         scan_polar_vectors = scan_polar_vectors[:, (scan_polar_vectors[1, :] <= angle_end) & 
# #                                                 (scan_polar_vectors[1, :] >= angle_start)]

# #         x_coords = ranges * np.cos(angles)
# #         y_coords = ranges * np.sin(angles)

# #         mask_estop = (np.abs(y_coords) <= self.car_width) & (x_coords <= self.estop_dist)
# #         close_points_count = np.sum(mask_estop)

# #         should_estop = bool(close_points_count >= self.count_threshold)

# #         shoe_found = Bool()
# #         shoe_found.data = should_estop  
# #         self.shoe_pub.publish(shoe_found)
    
# # def main(args=None):
# #     rclpy.init(args=args)
# #     cone_detector = PersonDetector()
# #     rclpy.spin(cone_detector)
# #     rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()
