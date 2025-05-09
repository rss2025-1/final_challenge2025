#!/usr/bin/env python3
"""
racing_controller.py

A controller for racing (lane following) using a pure pursuit controller with
added stabilization to reduce oscillations within the lane.
"""

import rclpy
from rclpy.node import Node
import math
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32  # Added for error publishing
from cv_bridge import CvBridge
import cv2

# Import the lane detector module.
import lane_follower.lane_detector as lane_detector


class RacingController(Node):
    def __init__(self):
        super().__init__('racing_controller')

        # Declare parameters.
        self.declare_parameter("drive_topic", "/vesc/low_level/input/navigation")
        self.declare_parameter("camera_topic", "/zed/zed_node/left_raw/image_raw_color")
        self.declare_parameter("error_topic", "/lane_error")  # New parameter for error topic
        self.declare_parameter("car_length", 0.325)          # meters
        self.declare_parameter("lookahead_distance", 2.5)    # meters, reduced from 1.0
        self.declare_parameter("max_speed", 3.0)             # maximum forward speed
        self.declare_parameter("road_width", 0.89)            # estimated road width in meters
        self.declare_parameter("lateral_filter_coeff", 0.7)  # Filter coefficient for lateral offset
        self.declare_parameter("steering_damping", 0.35)      # Damping coefficient for steering
        self.declare_parameter("steering_bias", -0.0035)       # Bias correction for mechanical veer (positive for right, negative for left)

        self.drive_topic = self.get_parameter("drive_topic").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.error_topic = self.get_parameter("error_topic").value  # Get error topic name
        self.car_length = self.get_parameter("car_length").value
        self.lookahead_distance = self.get_parameter("lookahead_distance").value
        self.max_speed = self.get_parameter("max_speed").value
        self.road_width = self.get_parameter("road_width").value
        self.lateral_filter_coeff = self.get_parameter("lateral_filter_coeff").value
        self.steering_damping = self.get_parameter("steering_damping").value
        self.steering_bias = self.get_parameter("steering_bias").value
        self.get_logger().info(self.drive_topic)
        self.get_logger().info(f"self.camera_topic: {self.camera_topic}")
        self.get_logger().info(f"self.error_topic: {self.error_topic}")  # Log error topic
        self.get_logger().info(f"self.car_length: {self.car_length}")
        self.get_logger().info(f"self.lookahead_distance: {self.lookahead_distance}")
        self.get_logger().info(f"self.max_speed: {self.max_speed}")
        self.get_logger().info(f"self.road_width: {self.road_width}")
        self.get_logger().info(f"self.lateral_filter_coeff: {self.lateral_filter_coeff}")
        self.get_logger().info(f"self.steering_damping: {self.steering_damping}")
        self.get_logger().info(f"self.steering_bias: {self.steering_bias}")


        # State variables for filtering
        self.prev_lateral_offset = 0.0
        self.prev_steering_angle = 0.0
        self.last_time = self.get_clock().now()

        # Publisher for drive commands.
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        
        # Publisher for error/offset data (for visualization)
        self.error_pub = self.create_publisher(Float32, self.error_topic, 10)
        
        # Subscription for camera images.
        self.create_subscription(Image, self.camera_topic, self.image_callback, 1)
        
        # CvBridge for converting ROS image msg to OpenCV image.
        self.bridge = CvBridge()

        self.get_logger().info("Racing Controller Initialized")

    def image_callback(self, msg):
        # self.get_logger().info("Image callback called.")
        # Calculate time delta for time-based control
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9  # Convert to seconds
        self.last_time = current_time

        # Convert the incoming ROS Image message to an OpenCV BGR image.
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # Convert BGR to RGB as the lane_detector expects an RGB image.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Process the image to detect lanes.
        output_img, x_position, theta = lane_detector.detect_lanes(rgb_image)
        
        # If lane detection fails, send zero command
        drive_cmd = AckermannDriveStamped()
        # if x_position is None:
        #     self.get_logger().debug("Lane detection failed; sending zero command.")
        #     drive_cmd.drive.speed = 0.0
        #     drive_cmd.drive.steering_angle = 0.0
        #     self.drive_pub.publish(drive_cmd)
        #     return
        
        if x_position is None:
            self.get_logger().info("Lane detection failed; sending slower drive straight command.")
            drive_cmd.drive.speed = self.max_speed
            drive_cmd.drive.steering_angle = -0.02
            self.drive_pub.publish(drive_cmd)
            return

        # Get image dimensions.
        img_height, img_width, _ = rgb_image.shape
        image_center = img_width / 2.0

        # Convert pixel error into an estimated real-world lateral offset.
        pixel_to_meter = self.road_width / img_width
        raw_lateral_offset = (x_position - image_center) * pixel_to_meter

        # Apply low-pass filter to lateral offset to reduce noise
        filtered_lateral_offset = (self.lateral_filter_coeff * self.prev_lateral_offset + 
                                  (1 - self.lateral_filter_coeff) * raw_lateral_offset)
        self.prev_lateral_offset = filtered_lateral_offset
        
        # Calculate lateral offset derivative (rate of change)
        lateral_offset_derivative = (filtered_lateral_offset - self.prev_lateral_offset) / max(dt, 0.001)

        # Publish the lateral offset error for visualization
        error_msg = Float32()
        error_msg.data = float(filtered_lateral_offset)
        self.error_pub.publish(error_msg)

        # Compute the angle to the target in the vehicle coordinate system.
        alpha = math.atan2(filtered_lateral_offset, self.lookahead_distance)

        # Pure pursuit steering law with added damping
        raw_steering_angle = math.atan2(2.0 * self.car_length * math.sin(alpha), self.lookahead_distance)
        
        # Apply damping to steering angle to reduce oscillations
        damped_steering_angle = ((1 - self.steering_damping) * raw_steering_angle + 
                                self.steering_damping * self.prev_steering_angle)
                                
        # Apply steering bias to correct for mechanical veer
        # Positive value compensates for leftward veer by adding right bias
        corrected_steering_angle = damped_steering_angle + self.steering_bias
        
        self.prev_steering_angle = corrected_steering_angle

        # Calculate a more sophisticated speed reduction based on steering angle and road conditions
        # Use quadratic relationship for smoother speed transitions
        steering_magnitude = abs(corrected_steering_angle)
        speed_reduction_factor = max(0.3, 1 - (steering_magnitude / (math.pi / 3))**2)
        
        # Reduce speed more when lateral error is changing rapidly (indicates instability)
        if abs(lateral_offset_derivative) > 0.5:  # Threshold for rapid change
            speed_reduction_factor *= 0.8  # Further reduce speed during rapid changes
            
        speed = self.max_speed * speed_reduction_factor

        # Populate and publish the drive command.
        self.get_logger().info(f"Speed: {speed}")
        self.get_logger().info(f"Steering Angle: {corrected_steering_angle}")
        drive_cmd.drive.speed = float(speed)
        drive_cmd.drive.steering_angle = float(corrected_steering_angle)

        # Log values for debugging
        self.get_logger().debug(
            f"Raw offset: {raw_lateral_offset:.3f} m, Filtered: {filtered_lateral_offset:.3f} m, "
            f"alpha: {alpha:.3f} rad, raw_steering: {raw_steering_angle:.3f}, "
            f"damped_steering: {damped_steering_angle:.3f}, corrected_steering: {corrected_steering_angle:.3f}, "
            f"speed: {speed:.3f} m/s"
        )

        self.drive_pub.publish(drive_cmd)

def main(args=None):
    rclpy.init(args=args)
    racing_controller = RacingController()
    rclpy.spin(racing_controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()