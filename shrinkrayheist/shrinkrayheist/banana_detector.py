#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from .model.detector import Detector #this is YOLO
from nav_msgs.msg import Odometry

class BananaDetector(Node):
    def __init__(self):
        super().__init__("banana_detector")

        self.banana_state_pub = self.create_publisher(Bool, "/banana_detected", 10)  
        self.debug_pub = self.create_publisher(Image, "/banana_debug_img", 10)
        
        self.odom_sub = self.create_subscription(Odometry,"/pf/pose/odom", self.pose_callback, 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)

        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.detector = Detector()  # Load YOLO model
        self.detector.set_threshold(0.5)  # Optional: you can adjust threshold

        self.get_logger().info("Banana Detector Initialized")

    def find_closest_banana(self, predictions):
        """
        Finds the closest banana from the predictions.
        """
        banana_detected = False
        closest_banana = None
        largest_banana_size_detected = 0

        for (x1, y1, x2, y2), label in predictions:
            if label.lower() == "banana":  # Assuming YOLO model uses label "banana"
                # Calculate center of bounding box
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                center = np.array([u, v])

                # Find the center of the closest banana detected
                size = (x2 - x1) * (y2 - y1)
                if size > largest_banana_size_detected:
                    largest_banana_size_detected = size
                    closest_banana = center
                banana_detected = True
        return banana_detected, closest_banana, largest_banana_size_detected

    def image_callback(self, image_msg):
        """
        Uses YOLO to detect banana. If detected, publish location and detection state.
        """
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Run YOLO detection
        results = self.detector.predict(image)
        predictions = results["predictions"]
        debug_image = results["original_image"]

        # Find the closest banana
        banana_detected, closest_banana, largest_banana_size_detected = self.find_closest_banana(predictions)

        # Publish the banana detection state
        banana_msg = Bool()
        banana_msg.data = banana_detected
        self.banana_state_pub.publish(banana_msg)

        if banana_detected and closest_banana is not None:
            self.get_logger().info(f"Closest banana detected at: {closest_banana}")
            self.get_logger().info(f"Largest banana size detected: {largest_banana_size_detected}")

        # Publish debug image (with boxes)
        pil_debug_image = self.detector.draw_box(debug_image, predictions, draw_all=True)
        debug_image = np.array(pil_debug_image)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        self.debug_pub.publish(debug_msg)

    def pose_callback(self, msg):
        """
        Currently unused. Could be used to implement banana_collected() logic.
        """
        pass

    def banana_collected(self):
        """
        TODO: Update this function based on robot pose and banana pixel position
        """
        pass

def main(args=None):
    rclpy.init(args=args)
    banana_detector = BananaDetector()
    rclpy.spin(banana_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
