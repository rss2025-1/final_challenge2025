#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

# ## Pixel measurements from the image plane
# PTS_IMAGE_PLANE = [[204.0, 256.0],
#                 [457.0, 256.0],
#                 [148.0, 219.0],
#                 [490.0, 216.0]] 
# ######################################################

# # PTS_GROUND_PLANE units are in inches
# # car looks along positive x axis with positive y axis to left

# ######################################################
# ## Real world measurements from the ground plane starting at the camera, using letter-sized paper
# PTS_GROUND_PLANE = [[11.0, 8.5],
#                     [11.0, -8.5],
#                     [22.0, 17.0],
#                     [22.0, -17.0]] 
# ######################################################

# METERS_PER_INCH = 0.0254

class SimpleLaneFollower(Node):
    """
    Simple lane follower that finds left and right lane lines and their intersection point.
    """
    def __init__(self):
        super().__init__("simple_lane_follower")
        
        # Parameters
        self.declare_parameter("max_speed", 4.0)  # m/s
        self.declare_parameter("steering_gain", 0.6)  # Proportional gain for steering
        self.declare_parameter("steering_derivative_gain", 0.2)  # Derivative gain for steering
        self.declare_parameter("lookahead_distance", 150)  # pixels
        
        # Get parameters
        self.max_speed = self.get_parameter("max_speed").value
        self.steering_gain = self.get_parameter("steering_gain").value
        self.steering_derivative_gain = self.get_parameter("steering_derivative_gain").value
        self.lookahead_distance = self.get_parameter("lookahead_distance").value
        
        # Image processing parameters
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.min_line_length = 30
        self.max_line_gap = 20
        self.min_slope = 0.3  # Minimum slope to consider a line (increased to filter out horizontal lines)
        self.max_slope = 10.0  # Maximum slope to filter out near-vertical lines
        self.white_threshold = 220  # Increased threshold for stricter white color detection
        
        # Define ROI vertices - wider trapezoid to focus on lane area
        # This is crucial for filtering out irrelevant edges in the image
        height, width = 480, 640  # Default camera resolution
        self.roi_vertices = np.array([
            [(0, height), (width//6, height//2), (5*width//6, height//2), (width, height)]
        ], dtype=np.int32)
        
        # Publishers and subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/navigation", 10)
        self.debug_pub = self.create_publisher(Image, "/lane_debug_img", 10)
        
        self.image_sub = self.create_subscription(
            Image,
            "/zed/zed_node/rgb/image_rect_color",
            self.image_callback,
            5
        )
        
        # Controller state
        self.prev_error = 0.0
        self.prev_steering_angle = 0.0
        
        # Speed control parameters
        self.min_speed = 1.0  # Minimum speed in m/s
        self.curve_speed_factor = 0.7  # Speed reduction factor for curves
        
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Pure Pursuit Controller
        # Initialize data into a homography matrix
        # np_pts_ground = np.array(PTS_GROUND_PLANE)
        # np_pts_ground = np_pts_ground * METERS_PER_INCH
        # np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        # np_pts_image = np.array(PTS_IMAGE_PLANE)
        # np_pts_image = np_pts_image * 1.0
        # np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        # self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)
        # self.get_logger().info("Homography Transformer Initialized")

        # self.look_ahead = 0.5 # Lookahead distance
        # self.wheelbase = 0.34 # Length of wheelbase
        # self.max_steer_angle = 0.34
        
        self.get_logger().info("Lane Follower with PD Controller Initialized")

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

        # Enhanced white color masking using HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for white color in HSV
        # Low saturation and high value (brightness) for white
        lower_white = np.array([0, 0, self.white_threshold])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white color
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((3,3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Apply Canny edge detection
        edges = cv2.Canny(white_mask, self.canny_low_threshold, self.canny_high_threshold)

        # Apply region of interest mask
        mask = np.zeros_like(edges)
        roi_vertices = np.array([
            [(0, edges.shape[0]), (width//6, 0), (5*width//6, 0), (width, edges.shape[0])]
        ], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        return lines, crop_height

    def find_lane_lines(self, lines):
        """
        Separate detected lines into left and right lane lines.

        Args:
            lines: Lines detected by Hough transform

        Returns:
            left_line: Line representing the left lane boundary
            right_line: Line representing the right lane boundary
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
            # Skip lines that are too horizontal (slope too small) or too vertical (slope too large)
            if length < self.min_line_length or abs(slope) < self.min_slope or abs(slope) > self.max_slope:
                continue

            # Classify as left or right based on slope
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

        # Find best left and right lines
        left_line = self.average_line(left_lines) if left_lines else None
        right_line = self.average_line(right_lines) if right_lines else None

        return left_line, right_line

    def average_line(self, lines):
        """
        Calculate the average line from a list of lines.

        Args:
            lines: List of lines

        Returns:
            Average line as [x1, y1, x2, y2]
        """
        if not lines:
            return None

        x1_sum = sum(line[0][0] for line in lines)
        y1_sum = sum(line[0][1] for line in lines)
        x2_sum = sum(line[0][2] for line in lines)
        y2_sum = sum(line[0][3] for line in lines)

        count = len(lines)
        x1 = int(x1_sum / count)
        y1 = int(y1_sum / count)
        x2 = int(x2_sum / count)
        y2 = int(y2_sum / count)

        return [x1, y1, x2, y2]

    def find_intersection_point(self, left_line, right_line):
        """
        Find the intersection point of two lines.

        Args:
            left_line: Left lane line [x1, y1, x2, y2]
            right_line: Right lane line [x1, y1, x2, y2]

        Returns:
            Intersection point (x, y) or None if lines are parallel
        """
        if left_line is None or right_line is None:
            return None

        # Extract points from lines
        x1, y1, x2, y2 = left_line
        x3, y3, x4, y4 = right_line

        # Calculate line equations: y = mx + b
        if x2 - x1 == 0 or x4 - x3 == 0:  # Avoid division by zero
            return None

        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)
        b1 = y1 - m1 * x1
        b2 = y3 - m2 * x3

        # Check if lines are parallel
        if m1 == m2:
            return None

        # Calculate intersection point
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1

        return (int(x_intersect), int(y_intersect))
        
    def calculate_steering_angle(self, goal_x, car_position_x, width):
        """
        Calculate steering angle using a PD controller.
        
        Args:
            goal_x: X-coordinate of the goal point
            car_position_x: X-coordinate of the car's position
            width: Width of the image
            
        Returns:
            steering_angle: Calculated steering angle
        """
        # Calculate error (deviation from center to goal)
        error = goal_x - car_position_x
        
        # Calculate derivative of error
        error_derivative = error - self.prev_error
        
        # PD control law
        steering_angle = (self.steering_gain * error / (width / 2)) + \
                         (self.steering_derivative_gain * error_derivative / (width / 2))
        
        # Update previous error
        self.prev_error = error
        
        # Clip to valid steering range (-0.4 to 0.4 radians)
        steering_angle = np.clip(steering_angle, -0.4, 0.4)
        
        return steering_angle

    ## Pure Pursuit 
    # def transformUvToXy(self, u, v):
    #     """
    #     u and v are pixel coordinates.
    #     The top left pixel is the origin, u axis increases to right, and v axis
    #     increases down.

    #     Returns a normal non-np 1x2 matrix of xy displacement vector from the
    #     camera to the point on the ground plane.
    #     Camera points along positive x axis and y axis increases to the left of
    #     the camera.

    #     Units are in meters.
    #     """
    #     homogeneous_point = np.array([[u], [v], [1]])
    #     xy = np.dot(self.h, homogeneous_point)
    #     scaling_factor = 1.0 / xy[2, 0]
    #     homogeneous_xy = xy * scaling_factor
    #     x = homogeneous_xy[0, 0]
    #     y = homogeneous_xy[1, 0]
    #     return x, y

    # def find_ref_point(self, cx, cy):
    #     """
    #     Compute a reference point at `look_ahead` distance in the direction of the cone.
    #     If the cone is closer than that, the reference point is the cone itself.
    #     """
    #     la_dist = self.look_ahead
    #     cone_pos = np.array([cx, cy])
    #     dist = np.sqrt(cx**2 + cy**2)
    #     if dist <= la_dist:
    #         return cone_pos, dist
    #     else:
    #         return ((cone_pos / dist) * la_dist, dist)

    # def calculate_steering_angle_pp(self, x, y):
    #     x, y = self.transformUvToXy(x, y) # Convert pixel coordinates to real world coordinates
    #     rx, ry = self.find_ref_point(x, y)
    #     eta = np.arctan2(ry, rx)
    #     L = self.wheelbase
    #     L1 = self.look_ahead
    #     steer_angle = np.arctan2(2*np.sin(eta)*L, L1)
    #     return steer_angle if steer_angle <= self.max_steer_angle else self.max_steer_angle
    
    def calculate_speed(self, steering_angle):
        """
        Calculate the appropriate speed based on steering angle.
        
        Args:
            steering_angle: Current steering angle
            
        Returns:
            speed: Calculated speed
        """
        # Base speed is maximum speed
        speed = self.max_speed
        
        # Reduce speed for sharp turns
        turn_factor = 1.0 - (abs(steering_angle) / 0.4) * (1.0 - self.curve_speed_factor)
        speed *= turn_factor
        
        # Ensure speed doesn't go below minimum
        return max(self.min_speed, min(speed, self.max_speed))

    def image_callback(self, image_msg):
        """
        Process incoming camera images to find lane lines and their intersection.

        Args:
            image_msg: ROS Image message
        """
        try:
            # Convert ROS Image to OpenCV image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            height, width = image.shape[:2]
            
            # Create a copy for visualization
            debug_img = image.copy()
            
            # Detect lane lines
            lines, crop_height = self.detect_lane_lines(image)
            
            # Find left and right lane lines
            left_line, right_line = self.find_lane_lines(lines)
            
            # Find intersection point
            intersection_point = self.find_intersection_point(left_line, right_line)
            
            # Visualize results
            if left_line is not None:
                x1, y1, x2, y2 = left_line
                # Adjust y-coordinates for the cropped image
                y1 += crop_height
                y2 += crop_height
                
                # Draw the detected left line segment
                cv2.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for left line
                cv2.putText(debug_img, "Left Line", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Calculate and draw extended left line
                if x2 - x1 != 0:  # Avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    # Extend to top of image
                    top_y = 0
                    top_x = int((top_y - intercept) / slope) if slope != 0 else x1
                    
                    # Extend to bottom of image
                    bottom_y = height
                    bottom_x = int((bottom_y - intercept) / slope) if slope != 0 else x1
                    
                    # Draw extended line (dashed)
                    cv2.line(debug_img, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 0), 1, cv2.LINE_AA)
            
            if right_line is not None:
                x1, y1, x2, y2 = right_line
                # Adjust y-coordinates for the cropped image
                y1 += crop_height
                y2 += crop_height
                
                # Draw the detected right line segment
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for right line
                cv2.putText(debug_img, "Right Line", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Calculate and draw extended right line
                if x2 - x1 != 0:  # Avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    # Extend to top of image
                    top_y = 0
                    top_x = int((top_y - intercept) / slope) if slope != 0 else x1
                    
                    # Extend to bottom of image
                    bottom_y = height
                    bottom_x = int((bottom_y - intercept) / slope) if slope != 0 else x1
                    
                    # Draw extended line (dashed)
                    cv2.line(debug_img, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 1, cv2.LINE_AA)
            
            if intersection_point is not None:
                x_intersect, y_intersect = intersection_point
                # Adjust y-coordinate for the cropped image
                y_intersect += crop_height
                
                # Draw the intersection point
                cv2.circle(debug_img, (x_intersect, y_intersect), 10, (255, 0, 255), -1)  # Magenta circle for intersection
                cv2.putText(debug_img, "Intersection", (x_intersect+10, y_intersect), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Calculate a closer goal point at a fixed distance from the car
                # The car is assumed to be at the bottom center of the image
                car_position = (width // 2, height)
                
                # Define a fixed lookahead distance (adjust this value as needed)
                lookahead_distance = 150  # pixels
                
                # Calculate the direction vector from car to intersection
                dir_x = x_intersect - car_position[0]
                dir_y = y_intersect - car_position[1]
                
                # Normalize the direction vector
                dir_length = np.sqrt(dir_x**2 + dir_y**2)
                if dir_length > 0:
                    dir_x /= dir_length
                    dir_y /= dir_length
                
                # Calculate the goal point at the fixed distance in the direction of the intersection
                goal_x = int(car_position[0] + dir_x * lookahead_distance)
                goal_y = int(car_position[1] + dir_y * lookahead_distance)
                
                # Ensure the goal point is within the image bounds
                goal_x = max(0, min(goal_x, width-1))
                goal_y = max(0, min(goal_y, height-1))
                
                # Draw the goal point
                cv2.circle(debug_img, (goal_x, goal_y), 10, (0, 255, 0), -1)  # Green circle for goal
                cv2.putText(debug_img, "Goal Point", (goal_x+10, goal_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw a line from car position to the goal point
                cv2.line(debug_img, car_position, (goal_x, goal_y), (0, 255, 255), 2)  # Yellow line
            
            # Calculate steering angle and speed if goal point is found
            if intersection_point is not None:
                # Use the closer goal point for steering
                steering_angle = self.calculate_steering_angle(goal_x, car_position[0], width)
                speed = self.calculate_speed(steering_angle)
                
                # Create and publish drive command
                drive_cmd = AckermannDriveStamped()
                drive_cmd.header.stamp = self.get_clock().now().to_msg()
                drive_cmd.drive.steering_angle = steering_angle
                drive_cmd.drive.speed = speed
                self.drive_pub.publish(drive_cmd)
                
                # Add status information with steering and speed
                status_text = f"Intersection: ({x_intersect}, {y_intersect}) | Steering: {steering_angle:.2f} | Speed: {speed:.1f} m/s"
                cv2.putText(debug_img, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(debug_img, "No intersection found - STOPPED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Publish stop command if no goal point
                drive_cmd = AckermannDriveStamped()
                drive_cmd.header.stamp = self.get_clock().now().to_msg()
                drive_cmd.drive.steering_angle = 0.0
                drive_cmd.drive.speed = 0.0
                self.drive_pub.publish(drive_cmd)
            
            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)
            
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            import traceback
            self.get_logger().error(f"Error in image_callback: {e}")
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")


def main(args=None):
    rclpy.init(args=args)
    lane_follower = SimpleLaneFollower()
    try:
        rclpy.spin(lane_follower)
    except KeyboardInterrupt:
        lane_follower.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        lane_follower.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
