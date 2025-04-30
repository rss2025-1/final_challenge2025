#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

class LaneFollower(Node):
    """
    Lane following node for the Race to the Moon challenge.
    This node detects lane lines using Hough transform, calculates the lane center,
    and controls the vehicle to stay within the assigned lane.
    """
    def __init__(self):
        super().__init__("lane_follower")

        # Parameters
        self.declare_parameter("lane_number", 3)
        self.declare_parameter("max_speed", 4.0)  # m/s
        self.declare_parameter("steering_p_gain", 0.5)  # Proportional gain
        self.declare_parameter("steering_d_gain", 0.3)  # Derivative gain

        self.lane_number = self.get_parameter("lane_number").value
        self.max_speed = self.get_parameter("max_speed").value
        self.steering_p_gain = self.get_parameter("steering_p_gain").value
        self.steering_d_gain = self.get_parameter("steering_d_gain").value

        
        # PD controller state
        self.prev_deviation = 0.0
        self.prev_time = None

        # Publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/navigation", 10)
        self.debug_pub = self.create_publisher(Image, "/lane_debug_img", 10)
        self.lane_departure_pub = self.create_publisher(Bool, "/lane_departure", 10)

        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback,
            5
        )

        # State variables
        self.outside_lane_start_time = None
        self.is_outside_lane = False
        self.lane_departure_count = 0
        self.long_breach_count = 0
        self.long_breach_reported = False  # Track if a long breach has been reported

        # Control parameters
        self.min_speed = 1.0  # Minimum speed in m/s
        self.curve_speed_factor = 0.7  # Speed reduction factor for curves
        self.recovery_speed_factor = 0.6  # Speed reduction factor during recovery

        # Image processing parameters
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.hough_threshold = 20
        self.min_line_length = 20
        self.max_line_gap = 300

        self.roi_vertices = None

        self.bridge = CvBridge()
        self.get_logger().info(f"Lane Follower Initialized for Lane {self.lane_number}")

    def detect_lane_lines(self, img):
        """
        Detect lane lines using Canny edge detection and Hough transform.

        Args:
            img: Input BGR image

        Returns:
            lines: Detected lines from Hough transform
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # white color mask
        lower_white = np.array([0,0,180])
        upper_white = np.array([180,60,255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        img = cv2.bitwise_and(img, img, mask=white_mask)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)

        # Focus on the region of interest (bottom portion of the image)
        mask = np.zeros_like(edges)
        height, width = edges.shape

        # Define a trapezoid region
        self.roi_vertices = np.array([
            [(0, height),
             (width//4, height//2),
             (3*width//4, height//2),
             (width, height)]
        ], dtype=np.int32)

        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        return lines, masked_edges

    def classify_lane_lines(self, lines, img_width):
        """
        Classify detected lines as left or right lane boundaries.

        Args:
            lines: Lines detected by Hough transform
            img_width: Width of the image

        Returns:
            left_lines: Lines classified as left lane boundaries
            right_lines: Lines classified as right lane boundaries
        """
        left_lines = []
        right_lines = []

        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter out horizontal lines
            if abs(slope) < 0.3:
                continue

            # Classify based on slope and position
            if slope < 0 and x1 < img_width // 2:
                left_lines.append(line)
            elif slope > 0 and x1 > img_width // 2:
                right_lines.append(line)

        return left_lines, right_lines

    def calculate_lane_center(self, left_lines, right_lines, img_shape):
        """
        Calculate the center of the lane based on detected lane lines.

        Args:
            left_lines: Lines classified as left lane boundaries
            right_lines: Lines classified as right lane boundaries
            img_shape: Shape of the image

        Returns:
            center_x: X-coordinate of the lane center
            bottom_y: Y-coordinate at the bottom of the image
            left_x: X-coordinate of the left lane boundary
            right_x: X-coordinate of the right lane boundary
        """
        height, width = img_shape[:2]

        # Default values if no lines are detected
        left_x = width // 4
        right_x = 3 * width // 4

        # Calculate average left line position at the bottom of the image
        if left_lines:
            left_points = []
            for line in left_lines:
                x1, y1, x2, y2 = line[0]
                # Extrapolate to bottom of image
                if y2 != y1:
                    left_x_at_bottom = int(x1 + (height - y1) * (x2 - x1) / (y2 - y1))
                    if 0 <= left_x_at_bottom < width:
                        left_points.append(left_x_at_bottom)

            if left_points:
                left_x = int(np.mean(left_points))

        # Calculate average right line position at the bottom of the image
        if right_lines:
            right_points = []
            for line in right_lines:
                x1, y1, x2, y2 = line[0]
                # Extrapolate to bottom of image
                if y2 != y1:
                    right_x_at_bottom = int(x1 + (height - y1) * (x2 - x1) / (y2 - y1))
                    if 0 < right_x_at_bottom <= width:
                        right_points.append(right_x_at_bottom)

            if right_points:
                right_x = int(np.mean(right_points))

        # Calculate center point
        center_x = (left_x + right_x) // 2

        return center_x, height - 1, left_x, right_x

    def detect_lane_departure(self, left_x, right_x, car_position_x):
        """
        Detect if the car is outside its assigned lane.

        Args:
            left_x: X-coordinate of the left lane boundary
            right_x: X-coordinate of the right lane boundary
            car_position_x: X-coordinate of the car's position

        Returns:
            is_outside: Boolean indicating if the car is outside the lane
        """
        # Check if car is outside lane boundaries
        if car_position_x < left_x or car_position_x > right_x:
            return True

        return False

    def calculate_steering_angle(self, center_x, car_position_x, img_width):
        """
        Calculate the steering angle using PD control based on the deviation from the lane center.

        Args:
            center_x: X-coordinate of the lane center
            car_position_x: X-coordinate of the car's position
            img_width: Width of the image

        Returns:
            steering_angle: Calculated steering angle
        """
        # Calculate deviation from lane center (normalized by image width)
        deviation = (center_x - car_position_x) / (img_width / 2)
        
        # Get current time for derivative calculation
        current_time = self.get_clock().now()
        
        # Calculate derivative term (rate of change of error)
        derivative = 0.0
        if self.prev_time is not None:
            # Calculate time delta in seconds
            dt = (current_time - self.prev_time).nanoseconds / 1e9
            if dt > 0:
                derivative = (deviation - self.prev_deviation) / dt
        
        # Store current values for next iteration
        self.prev_deviation = deviation
        self.prev_time = current_time
        
        # PD control: proportional term + derivative term
        steering_angle = self.steering_p_gain * deviation + self.steering_d_gain * derivative
        
        # Clip to valid steering range
        return np.clip(steering_angle, -0.4, 0.4)

    def calculate_speed(self, steering_angle, is_outside_lane):
        """
        Calculate the appropriate speed based on steering angle and lane position.

        Args:
            steering_angle: Current steering angle
            is_outside_lane: Boolean indicating if the car is outside the lane

        Returns:
            speed: Calculated speed
        """
        # Base speed is maximum speed
        speed = self.max_speed

        # Reduce speed for sharp turns
        turn_factor = 1.0 - (abs(steering_angle) / 0.4) * (1.0 - self.curve_speed_factor)
        speed *= turn_factor

        # Reduce speed when outside lane
        if is_outside_lane:
            speed *= self.recovery_speed_factor

        # Ensure speed doesn't go below minimum
        return max(self.min_speed, min(speed, self.max_speed))

    def image_callback(self, image_msg):
        """
        Process incoming camera images and control the vehicle.

        Args:
            image_msg: ROS Image message
        """
        try:
            # Convert ROS image to OpenCV format
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            height, width = image.shape[:2]

            # Detect lane lines
            lines, edges = self.detect_lane_lines(image)

            # Classify lines as left or right
            left_lines, right_lines = self.classify_lane_lines(lines, width)

            # Calculate lane center
            center_x, bottom_y, left_x, right_x = self.calculate_lane_center(
                left_lines, right_lines, image.shape
            )

            # Car position (assumed to be at the bottom center of the image)
            car_position_x = width // 2

            # Calculate steering angle
            steering_angle = self.calculate_steering_angle(center_x, car_position_x, width)

            # Check for lane departure
            current_outside_lane = self.detect_lane_departure(left_x, right_x, car_position_x)

            # Track lane departure
            current_time = self.get_clock().now()
            if current_outside_lane and not self.is_outside_lane:
                # New lane departure
                self.outside_lane_start_time = current_time
                self.is_outside_lane = True
                self.lane_departure_count += 1

                # Publish lane departure event
                departure_msg = Bool()
                departure_msg.data = True
                self.lane_departure_pub.publish(departure_msg)

                self.get_logger().warn("Lane departure detected!")

            elif current_outside_lane and self.is_outside_lane:
                # Continuing lane departure
                if self.outside_lane_start_time is not None:
                    duration = (current_time - self.outside_lane_start_time).nanoseconds / 1e9
                    if duration > 3.0:  # Long breach (>3 seconds)
                        if not hasattr(self, 'long_breach_reported') or not self.long_breach_reported:
                            self.long_breach_count += 1
                            self.long_breach_reported = True
                            self.get_logger().error(f"Long lane breach detected! (#{self.long_breach_count})")

            elif not current_outside_lane and self.is_outside_lane:
                # Recovered from lane departure
                self.outside_lane_start_time = None
                self.is_outside_lane = False
                self.long_breach_reported = False

                # Publish lane recovery event
                departure_msg = Bool()
                departure_msg.data = False
                self.lane_departure_pub.publish(departure_msg)

                self.get_logger().info("Recovered from lane departure")

            # Calculate appropriate speed
            speed = self.calculate_speed(steering_angle, self.is_outside_lane)

            # Create and publish drive command
            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.stamp = current_time.to_msg()
            drive_cmd.drive.steering_angle = steering_angle
            drive_cmd.drive.speed = speed
            self.drive_pub.publish(drive_cmd)

            # Log control values periodically (every 20 frames)
            if not hasattr(self, 'log_counter'):
                self.log_counter = 0
                
            self.log_counter += 1
            if self.log_counter % 20 == 0:
                self.get_logger().info(
                    f"Speed: {speed:.2f} m/s, Steering: {steering_angle:.2f}, "
                    f"Lane departures: {self.lane_departure_count}, "
                    f"Long breaches: {self.long_breach_count}"
                )

            # Create debug image
            debug_img = image.copy()

            # Draw lane lines
            if left_lines is not None:
                for line in left_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if right_lines is not None:
                for line in right_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw lane center and car position
            cv2.circle(debug_img, (center_x, bottom_y), 10, (0, 255, 0), -1)
            cv2.circle(debug_img, (car_position_x, bottom_y), 10, (255, 255, 0), -1)
            
            # Draw midpoint calculation line and point
            cv2.line(debug_img, (left_x, bottom_y), (right_x, bottom_y), (255, 0, 255), 2)  # Purple horizontal line
            
            # Draw the midpoint as a distinct marker (purple diamond)
            midpoint = (left_x + right_x) // 2
            diamond_size = 8
            diamond_points = np.array([
                [midpoint, bottom_y - diamond_size],  # top
                [midpoint + diamond_size, bottom_y],  # right
                [midpoint, bottom_y + diamond_size],  # bottom
                [midpoint - diamond_size, bottom_y]   # left
            ], np.int32)
            cv2.fillPoly(debug_img, [diamond_points], (255, 0, 255))  # Purple diamond
            
            cv2.putText(debug_img, f"Mid: {midpoint}", (midpoint - 40, bottom_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Draw lane boundaries
            cv2.line(debug_img, (left_x, bottom_y), (left_x, bottom_y - 50), (0, 255, 255), 3)
            cv2.line(debug_img, (right_x, bottom_y), (right_x, bottom_y - 50), (0, 255, 255), 3)

            # Add text with current status
            status_text = f"Lane: {self.lane_number}, Speed: {speed:.1f} m/s"
            cv2.putText(debug_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            departure_text = "LANE DEPARTURE!" if self.is_outside_lane else "In Lane"
            color = (0, 0, 255) if self.is_outside_lane else (0, 255, 0)
            cv2.putText(debug_img, departure_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # trapezoid vis
            cv2.polylines(debug_img, self.roi_vertices, isClosed=True, color=(0,255,255), thickness=2)
            # Publish debug image

            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    lane_follower = LaneFollower()
    rclpy.spin(lane_follower)
    lane_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()