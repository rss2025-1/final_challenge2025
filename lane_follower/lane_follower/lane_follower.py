#!/usr/bin/env python3

import rclpy
#testing
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

class LaneFollower(Node):
    """
    Lane follower node that implements a simplified computer vision pipeline
    to detect lane lines and control the vehicle steering.
    """
    def __init__(self):
        super().__init__("lane_follower")

        # Parameters
        self.declare_parameter("max_speed", 4.0)  # m/s
        self.declare_parameter("const_steering_angle", 0.036)  # rad - corrects natural drift
        self.declare_parameter("angle_bound", 0.2)  # rad - max steering angle
        self.declare_parameter("prev_angle_threshold", 0.1)  # rad - threshold for using prev angle
        self.declare_parameter("prev_angle_weight", 0.21)  # weight for previous angle in timer callback

        # Get parameters
        self.max_speed = self.get_parameter("max_speed").value
        self.const_steering_angle = self.get_parameter("const_steering_angle").value
        self.angle_bound = self.get_parameter("angle_bound").value
        self.prev_angle_threshold = self.get_parameter("prev_angle_threshold").value
        self.prev_angle_weight = self.get_parameter("prev_angle_weight").value

        # Publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/navigation", 10)
        self.debug_pub = self.create_publisher(Image, "/lane_debug_img", 10)

        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback,
            5
        )

        # Timer for intermediate drive commands (15 Hz)
        timer_period = 1/15  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # State variables
        self.prev_angle = self.const_steering_angle
        self.bridge = CvBridge()

        # Image processing parameters
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.min_line_length = 30  # minimum line length in pixels
        self.min_slope = 0.2  # minimum absolute slope value

        # State tracking variables
        self.lane_departure_count = 0
        self.long_breach_count = 0
        self.is_outside_lane = False
        self.lane_number = 1
        
        # Define ROI vertices for visualization
        height, width = 480, 640  # Default camera resolution
        self.roi_vertices = np.array([
            [(0, height), (width//3, height//2), (2*width//3, height//2), (width, height)]
        ], dtype=np.int32)

        self.get_logger().info("Lane Follower Initialized")

    def detect_lane_lines(self, img):
        """
        Detect lane lines using a simplified computer vision pipeline.

        Args:
            img: Input BGR image

        Returns:
            lines: Detected lines from Hough transform
            crop_height: Height where image was cropped
        """
        # Crop image to remove background features
        height, width = img.shape[:2]
        crop_height = height // 2
        img = img[crop_height:, :]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply white color segmentation
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Apply Canny edge detection
        edges = cv2.Canny(white_mask, self.canny_low_threshold, self.canny_high_threshold)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=self.min_line_length,
            maxLineGap=10
        )

        return lines, crop_height

    def find_best_line_pair(self, lines):
        """
        Find the best pair of non-intersecting lines for lane boundaries.

        Args:
            lines: Lines detected by Hough transform

        Returns:
            best_left: Best line for left boundary
            best_right: Best line for right boundary
        """
        if lines is None:
            return None, None

        left_lines = []
        right_lines = []

        # Filter and classify lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Skip vertical lines
                continue

            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Filter by length and slope
            if length < self.min_line_length or abs(slope) < self.min_slope:
                continue

            # Classify as left or right based on slope
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

        # Handle adversarial cases
        if not left_lines and not right_lines:
            return None, None
        elif not left_lines:
            # Case 1: Only right lines
            best_right = max(right_lines, key=lambda x: abs((x[0][3] - x[0][1])/(x[0][2] - x[0][0])))
            return None, best_right
        elif not right_lines:
            # Case 1: Only left lines
            best_left = max(left_lines, key=lambda x: abs((x[0][3] - x[0][1])/(x[0][2] - x[0][0])))
            return best_left, None

        # Find best non-intersecting pair
        best_pair = None
        min_intersection_y = float('-inf')

        for left in left_lines:
            for right in right_lines:
                # Calculate intersection point
                x1, y1, x2, y2 = left[0]
                x3, y3, x4, y4 = right[0]

                # Line equations: y = mx + b
                m1 = (y2 - y1) / (x2 - x1)
                m2 = (y4 - y3) / (x4 - x3)
                b1 = y1 - m1 * x1
                b2 = y3 - m2 * x3

                # Intersection x = (b2-b1)/(m1-m2)
                if m1 != m2:
                    x_intersect = (b2 - b1) / (m1 - m2)
                    y_intersect = m1 * x_intersect + b1

                    if y_intersect > min_intersection_y:
                        min_intersection_y = y_intersect
                        best_pair = (left, right)

        if best_pair:
            # Return the two lines separately, not as a tuple
            return best_pair[0], best_pair[1]
        else:
            # If no good pairs found, return steepest lines
            best_left = max(left_lines, key=lambda x: abs((x[0][3] - x[0][1])/(x[0][2] - x[0][0])))
            best_right = max(right_lines, key=lambda x: abs((x[0][3] - x[0][1])/(x[0][2] - x[0][0])))
            return best_left, best_right

    def calculate_goal_point(self, best_left, best_right, img_shape, crop_height):
        """
        Calculate the goal point based on lane lines.

        Args:
            best_left: Best line for left boundary
            best_right: Best line for right boundary
            img_shape: Shape of the image
            crop_height: Height where image was cropped

        Returns:
            goal_x: X-coordinate of the goal point
            left_x: X-coordinate of the left lane boundary
            right_x: X-coordinate of the right lane boundary
        """
        # Handle both 2-value and 3-value shape tuples
        if len(img_shape) == 3:
            height, width, _ = img_shape
        else:
            height, width = img_shape
        top_y = crop_height  # Top of the cropped image

        # Default values at image center
        left_x = width // 4
        right_x = 3 * width // 4
        goal_x = width // 2

        # Calculate intersection points with top of cropped image
        if best_left is not None:
            x1, y1, x2, y2 = best_left[0]
            if x2 - x1 != 0:  # Non-vertical line
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                left_x = int((top_y - intercept) / slope)

        if best_right is not None:
            x1, y1, x2, y2 = best_right[0]
            if x2 - x1 != 0:  # Non-vertical line
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                right_x = int((top_y - intercept) / slope)

        # Handle adversarial cases
        if best_left is None and best_right is not None:
            # Only right line visible - offset left by fixed amount
            goal_x = right_x - 20
        elif best_left is not None and best_right is None:
            # Only left line visible - offset right by fixed amount
            goal_x = left_x + 20
        else:
            # Both lines visible - use midpoint
            goal_x = (left_x + right_x) // 2

        # Ensure goal point is within image bounds
        goal_x = max(0, min(goal_x, width))

        return goal_x, left_x, right_x

    def calculate_steering_angle(self, goal_x, img_width):
        """
        Calculate steering angle based on the goal point.

        Args:
            goal_x: X-coordinate of the goal point
            img_width: Width of the image

        Returns:
            steering_angle: Calculated steering angle in radians
        """
        # Calculate dx (difference between goal and image center)
        dx = goal_x - (img_width // 2)
        
        # Use fixed dy (lookahead distance)
        dy = 160

        # Calculate angle using arctan
        angle = np.arctan2(dx, dy)

        # Map angle to bounded range and add constant steering correction
        steering_angle = np.interp(angle, [-np.pi, np.pi], [-self.angle_bound, self.angle_bound])
        steering_angle += self.const_steering_angle

        return steering_angle

    def timer_callback(self):
        """
        Timer callback for publishing intermediate drive commands.
        """
        # Calculate steering angle based on previous angle and constant correction
        if abs(self.prev_angle - self.const_steering_angle) > self.prev_angle_threshold:
            steering_angle = (self.prev_angle_weight * self.prev_angle + 
                            (1 - self.prev_angle_weight) * self.const_steering_angle)
        else:
            steering_angle = self.const_steering_angle

        # Create and publish drive command
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.drive.steering_angle = steering_angle
        drive_cmd.drive.speed = self.max_speed
        self.drive_pub.publish(drive_cmd)

    def image_callback(self, image_msg):
        """
        Process incoming camera images and control the vehicle.

        Args:
            image_msg: ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            height, width = image.shape[:2]

            # Default values in case no lines are detected
            goal_x = width // 2
            left_x = width // 4
            right_x = 3 * width // 4
            steering_angle = self.const_steering_angle
            
            # Detect lane lines
            lines, crop_height = self.detect_lane_lines(image)

            # Process detected lines if any were found
            if lines is not None:
                # Find best pair of lines
                best_left, best_right = self.find_best_line_pair(lines)

                # Calculate goal point
                goal_x, left_x, right_x = self.calculate_goal_point(
                    best_left, best_right, image.shape, crop_height
                )

                # Calculate steering angle
                steering_angle = self.calculate_steering_angle(goal_x, width)

            # Update previous angle for timer callback
            self.prev_angle = steering_angle

            # Car position (assumed to be at the bottom center of the image)
            car_position_x = width // 2
            bottom_y = height - 1

            # Check for lane departure
            self.lane_departure_detection(left_x, right_x, goal_x, car_position_x, bottom_y)

            # Adjust speed based on lane departure
            speed = self.max_speed * 0.8 if self.is_outside_lane else self.max_speed
            
            # Create debug image
            debug_img = image.copy()
            
            # Draw detected lane lines
            if lines is not None:
                # Draw all detected lines in light gray
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Adjust y coordinates for cropping
                    cv2.line(debug_img, (x1, y1 + crop_height), (x2, y2 + crop_height), (200, 200, 200), 1)
                
                # Draw best left line in red
                if best_left is not None:
                    x1, y1, x2, y2 = best_left[0]
                    # Adjust y coordinates for cropping
                    cv2.line(debug_img, (x1, y1 + crop_height), (x2, y2 + crop_height), (0, 0, 255), 2)
                    cv2.putText(debug_img, "Left", (x1, y1 + crop_height - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw best right line in blue
                if best_right is not None:
                    x1, y1, x2, y2 = best_right[0]
                    # Adjust y coordinates for cropping
                    cv2.line(debug_img, (x1, y1 + crop_height), (x2, y2 + crop_height), (255, 0, 0), 2)
                    cv2.putText(debug_img, "Right", (x1, y1 + crop_height - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw lane boundaries at the bottom of the image
            cv2.line(debug_img, (left_x, bottom_y), (left_x, bottom_y - 50), (0, 255, 255), 2)
            cv2.line(debug_img, (right_x, bottom_y), (right_x, bottom_y - 50), (0, 255, 255), 2)
            
            # Draw horizontal line connecting lane boundaries
            cv2.line(debug_img, (left_x, bottom_y), (right_x, bottom_y), (255, 0, 255), 2)
            
            # Draw midpoint as a distinct marker
            midpoint = (left_x + right_x) // 2
            cv2.circle(debug_img, (midpoint, bottom_y), 5, (255, 0, 255), -1)
            cv2.putText(debug_img, f"Mid: {midpoint}", (midpoint - 40, bottom_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            
            # Draw goal point
            cv2.circle(debug_img, (goal_x, crop_height), 8, (0, 255, 0), -1)
            cv2.putText(debug_img, "Goal", (goal_x + 10, crop_height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Draw car position
            cv2.circle(debug_img, (car_position_x, bottom_y), 8, (255, 255, 0), -1)
            cv2.putText(debug_img, "Car", (car_position_x + 10, bottom_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Draw steering direction line
            steer_length = 50
            steer_x = int(car_position_x + steer_length * np.sin(steering_angle))
            steer_y = int(bottom_y - steer_length * np.cos(steering_angle))
            cv2.line(debug_img, (car_position_x, bottom_y), (steer_x, steer_y), (0, 255, 255), 2)
            
            # Draw ROI trapezoid
            roi_vertices = np.array([
                [(0, height), (width//3, height//2), (2*width//3, height//2), (width, height)]
            ], dtype=np.int32)
            cv2.polylines(debug_img, roi_vertices, isClosed=True, color=(0, 255, 255), thickness=2)
            
            # Add status text
            status_text = f"Lane: {self.lane_number}, Speed: {speed:.1f} m/s, Angle: {steering_angle:.2f}"
            cv2.putText(debug_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add lane departure status
            departure_text = "LANE DEPARTURE!" if self.is_outside_lane else "In Lane"
            color = (0, 0, 255) if self.is_outside_lane else (0, 255, 0)
            cv2.putText(debug_img, departure_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)
            
            # Create and publish drive command
            current_time = self.get_clock().now()
            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.stamp = current_time.to_msg()
            drive_cmd.drive.steering_angle = steering_angle
            drive_cmd.drive.speed = speed
            self.drive_pub.publish(drive_cmd)
            
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            import traceback
            self.get_logger().error(f"Error in image_callback: {e}")
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def lane_departure_detection(self, left_x, right_x, center_x, car_position_x, bottom_y):
        """
        Detect and handle lane departures.

        Args:
            left_x: X-coordinate of left lane boundary
            right_x: X-coordinate of right lane boundary
            center_x: X-coordinate of lane center
            car_position_x: X-coordinate of car position
            bottom_y: Y-coordinate for visualization
        """
        # Initialize variables
        if not hasattr(self, 'outside_lane_start_time'):
            self.outside_lane_start_time = None
            self.long_breach_reported = False

        # Check if car is outside lane boundaries
        current_outside_lane = not (left_x < car_position_x < right_x)

        # Handle lane departure state changes
        if current_outside_lane and not self.is_outside_lane:
            # New lane departure
            self.is_outside_lane = True
            self.outside_lane_start_time = self.get_clock().now()
            self.lane_departure_count += 1
            self.get_logger().warn("Lane departure detected!")

        elif current_outside_lane and self.is_outside_lane:
            # Continuing lane departure
            if self.outside_lane_start_time is not None:
                duration = (self.get_clock().now() - self.outside_lane_start_time).nanoseconds / 1e9
                if duration > 3.0 and not self.long_breach_reported:  # Long breach (>3 seconds)
                    self.long_breach_count += 1
                    self.long_breach_reported = True
                    self.get_logger().error(f"Long lane breach detected! (#{self.long_breach_count})")

        elif not current_outside_lane and self.is_outside_lane:
            # Recovered from lane departure
            self.is_outside_lane = False
            self.outside_lane_start_time = None
            self.long_breach_reported = False
            self.get_logger().info("Recovered from lane departure")


def main(args=None):
    rclpy.init(args=args)
    lane_follower = LaneFollower()
    try:
        rclpy.spin(lane_follower)
    except KeyboardInterrupt:
        lane_follower.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        lane_follower.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
