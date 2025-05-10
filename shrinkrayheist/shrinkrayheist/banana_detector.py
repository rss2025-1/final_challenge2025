#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from .model.detector import Detector #this is YOLO
import os
from ackermann_msgs.msg import AckermannDriveStamped
import time

class BananaDetector(Node):
    def __init__(self):
        super().__init__("banana_detector")

        self.banana_state_pub = self.create_publisher(Bool, "/banana_detected", 10)  
        self.debug_pub = self.create_publisher(Image, "/banana_debug_img", 10)
        
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)

        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.detector = Detector()  # Load YOLO model
        self.detector.set_threshold(0.5)  # Optional: you can adjust threshold

        # State flags
        self._detection_enabled = True
        self._detection_pause_time = 40.0
        self._phase1_timer = None
        self._phase2_timer = None
        self.screenshot_enabled = True

        self.banana_count = 0

        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/safety", 10)
        self.goal_reached_pub = self.create_publisher(Bool, "/goal_reached", 10)

        self.get_logger().info("Banana Detector Initialized")

    def find_closest_banana(self, predictions):
        """
        Finds the closest banana from the predictions.
        """
        banana_detected = False
        closest_banana = None
        largest_banana_size_detected = 0

        for (x1, y1, x2, y2), label in predictions:
            if label.lower() in {"banana", "frisbee"}:  # Assuming YOLO model uses label "banana"
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

    def publish_debug_image(self, debug_image, predictions):
        # Publish debug image (with boxes)
        pil_debug_image = self.detector.draw_box(debug_image, predictions, draw_all=True)
        debug_image = np.array(pil_debug_image)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        self.debug_pub.publish(debug_msg)

    def screenshot_banana(self, debug_image, predictions):
        # Screenshot debug image (with boxes)
        pil_debug_image = self.detector.draw_box(debug_image, predictions, draw_all=True)
        debug_image = np.array(pil_debug_image)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(script_dir, "banana_img")
        fname = os.path.join(save_dir, f"banana_{self.banana_count}.png")
        self.banana_count += 1
        success = cv2.imwrite(fname, debug_image)
        if success:
            self.get_logger().info(f"Saved banana image to {fname}")
        else:
            self.get_logger().error(f"Failed to save banana image to {fname}")

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

        if not self._detection_enabled:
            # Pause detection after stopping for banana, but still publishes debugging image
            msg = Bool(data=False)
            self.banana_state_pub.publish(msg)
            self.publish_debug_image(debug_image, predictions)
            return

        # Find the closest banana
        banana_detected, closest_banana, largest_banana_size_detected = self.find_closest_banana(predictions)

        # Publish the banana detection state
        banana_msg = Bool(data = banana_detected)
        if largest_banana_size_detected > 2000.0:
            self.banana_state_pub.publish(banana_msg)

        if banana_detected and closest_banana is not None:
            # self.get_logger().info(f"Closest banana detected at: {closest_banana}")
            # self.get_logger().info(f"Largest banana size detected: {largest_banana_size_detected}")
            pass

        # Stop when the banana is found
        if banana_detected and self._detection_enabled:
            self.get_logger().info(f"Banana! stopping car and pausing detection.")
            self.goal_reached_pub.publish(Bool(data=bool(True)))
            if self.screenshot_enabled:
                self.screenshot_banana(debug_image, predictions)
                self.screenshot_enabled = False
            self._trigger_stop_and_pause()

        # Publish debug image (with boxes)
        self.publish_debug_image(debug_image, predictions)

    def _trigger_stop_and_pause(self):
        # Schedule Phase 1 timer: after 5 s, enter pause phase
        if self._phase1_timer is None:
            self._phase1_timer = self.create_timer(
                5.0,
                self._enter_pause_phase
            )

    def _enter_pause_phase(self):
        # Disable further detection
        self._detection_enabled = False
        self.get_logger().info(f"Detection disabled for {self._detection_pause_time}s")

        # Cancel Phase 1 timer so it only fires once
        if self._phase1_timer:
            self._phase1_timer.cancel()
            self._phase1_timer = None

        # Schedule Phase 2 timer: after 40 s of pause, resume detection
        if self._phase2_timer is None:
            self._phase2_timer = self.create_timer(
                self._detection_pause_time,
                self._resume_detection
            )
        
        # Reverse the car at 1.0 m/s for 5 seconds
        start = time.time()
        while time.time() - start < 3.0:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.speed = -1.0
            self.drive_pub.publish(drive_msg)


    def _resume_detection(self):
        # Re-enable detection
        self._detection_enabled = True
        self.screenshot_enabled = True

        # Cancel Phase 2 timer
        if self._phase2_timer:
            self._phase2_timer.cancel()
            self._phase2_timer = None

def main(args=None):
    rclpy.init(args=args)
    banana_detector = BananaDetector()
    rclpy.spin(banana_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
