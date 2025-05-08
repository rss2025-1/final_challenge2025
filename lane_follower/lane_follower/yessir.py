import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped

## Pixel measurements from the image plane
PTS_IMAGE_PLANE = [[204.0, 256.0],
                [457.0, 256.0],
                [148.0, 219.0],
                [490.0, 216.0]] 
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
## Real world measurements from the ground plane starting at the camera, using letter-sized paper
PTS_GROUND_PLANE = [[11.0, 8.5],
                    [11.0, -8.5],
                    [22.0, 17.0],
                    [22.0, -17.0]] 
######################################################

METERS_PER_INCH = 0.0254

class SimpleLaneFollower(Node):
    """
    Simple lane follower that finds left and right lane lines and their intersection point.
    """
    def __init__(self):
        super().__init__("simple_lane_follower")
        
        # Parameters
        self.declare_parameter("max_speed", 4.0)  # m/s
        self.declare_parameter("steering_gain", 0.4)  # Proportional gain for steering
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
        self.min_slope = 0.1  # Minimum slope to consider a line (increased to filter out horizontal lines)
        self.max_slope = 10.0  # Maximum slope to filter out near-vertical lines
        self.white_threshold = 200  # Increased threshold for stricter white color detection
        self.crop_height = None
        
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
        
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Pure Pursuit Controller
        # Initialize data into a homography matrix
        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)
        self.get_logger().info("Homography Transformer Initialized")

        self.look_ahead = 2.5  # Lookahead distance in meters
        self.wheelbase = 0.34  # Length of wheelbase in meters
        self.max_steer_angle = 0.34  # Maximum steering angle in radians
        
        self.get_logger().info("Lane Follower with PD Controller Initialized")

    def detect_lane_lines(self, img):
        """
        Detect lane lines using a simplified computer vision pipeline.

        Args:
            img: Input BGR image

        Returns:
            lines: Detected lines from Hough transform
            crop_height: Height where image was cropped
            debug_img: Debug image with visualization (optional)
        """
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Crop image to remove background features
        crop_height = height // 2
        self.crop_height = height // 2
        img = img[self.crop_height:, :]

        # Enhanced white color masking using HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for white color in HSV
        # Low saturation and high value (brightness) for white
        lower_white = np.array([0, 0, self.white_threshold])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white color
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply more aggressive morphological operations to improve robustness
        kernel = np.ones((3, 3), np.uint8)
        
        # First erode to remove small noise
        white_mask = cv2.erode(white_mask, kernel, iterations=1)
        
        # Then dilate to enhance the remaining white pixels (lane lines)
        white_mask = cv2.dilate(white_mask, kernel, iterations=2)
        
        # Apply morphological closing to fill gaps in lane lines
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create a debug image for visualization
        debug_mask = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

        # Apply Canny edge detection on the processed mask
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

        return lines, crop_height, debug_mask

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
        y1 = int(y1_sum / count) + self.crop_height
        x2 = int(x2_sum / count)
        y2 = int(y2_sum / count) + self.crop_height

        return [x1, y1, x2, y2]

    def average_center_line(self, left_line, right_line):
        """
        Compute the line exactly midway between two lane boundaries,
        and extend it from y=0 (top of image) to y=image_height-1 (bottom of image).

        Args:
            left_line:  [x1, y1, x2, y2] for the left lane (or None)
            right_line: [x1, y1, x2, y2] for the right lane (or None)

        Returns:
            extended_center_line: [x_at_top, 0, x_at_bottom, image_height-1] 
                                  representing the extended center line.
                                  Returns None if input lines are missing,
                                  if the calculated center line is horizontal (cannot extend),
                                  or if image_height is invalid.
        """
        if left_line is None or right_line is None:
            return None, [240, 320]

        # Calculate the midpoints of the segments forming the center line
        x1_mid = int((left_line[0] + right_line[0]) / 2.0)
        y1_mid = int((left_line[1] + right_line[1]) / 2.0)
        x2_mid = int((left_line[2] + right_line[2]) / 2.0)
        y2_mid = int((left_line[3] + right_line[3]) / 2.0)

        # Extend the center line to the top and bottom of the image
        y_top_screen = 0
        y_bottom_screen = 480

        # Case 1: Vertical line (x1_mid == x2_mid)
        if x1_mid == x2_mid:
            # The line is vertical, x-coordinate is constant.
            return [x1_mid, y_top_screen, x1_mid, y_bottom_screen], [240, 320]

        slope = (y2_mid - y1_mid) / float(x2_mid - x1_mid)
        intercept = y1_mid - slope * x1_mid
        x_at_top = int(round((y_top_screen - intercept) / slope))
        x_at_bottom = int(round((y_bottom_screen - intercept) / slope))

        # Point followed by the controller
        midpoint_x = int((x_at_top + x_at_bottom) / 2.0)    
        midpoint_y = int((y_top_screen + y_bottom_screen) / 2.0)
    
        return [x_at_top, y_top_screen, x_at_bottom, y_bottom_screen], [midpoint_x, midpoint_y]


####################################################################################################################################################################################
    ## Pure Pursuit 
    def transformUvToXy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y

    def find_ref_point(self, cx, cy):
        """
        Compute a reference point at `look_ahead` distance in the direction of the cone.
        If the cone is closer than that, the reference point is the cone itself.
        """
        la_dist = self.look_ahead
        cone_pos = np.array([cx, cy])
        dist = np.sqrt(cx**2 + cy**2)
        if dist <= la_dist:
            return cone_pos, dist
        else:
            return ((cone_pos / dist) * la_dist, dist)

    def calculate_steering_angle_pp(self, x, y):
        """
        Calculate steering angle using Pure Pursuit controller.
        
        Args:
            x: X-coordinate of the goal point in pixel coordinates
            y: Y-coordinate of the goal point in pixel coordinates
            
        Returns:
            steering_angle: Calculated steering angle using Pure Pursuit
        """
        x, y = self.transformUvToXy(x, y)  # Convert pixel coordinates to real world coordinates
        rx, ry = self.find_ref_point(x, y)
        eta = np.arctan2(ry, rx)
        L = self.wheelbase
        L1 = self.look_ahead
        steer_angle = np.arctan2(2*np.sin(eta)*L, L1)
        return steer_angle if abs(steer_angle) <= self.max_steer_angle else (self.max_steer_angle if steer_angle > 0 else -self.max_steer_angle)

    def drive(self, steering_angle, speed):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = self.get_clock().now().to_msg()
        drive_cmd.drive.steering_angle = steering_angle
        drive_cmd.drive.speed = speed
        self.drive_pub.publish(drive_cmd)

####################################################################################################################################################################################

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
            
            # Detect lane lines
            lines, crop_height, mask_debug = self.detect_lane_lines(image)
            left_line, right_line = self.find_lane_lines(lines)
            center_line, goal_point = self.average_center_line(left_line, right_line)

            # Motion Controller
            steering_angle = self.calculate_steering_angle_pp(goal_point[0], goal_point[1])
            speed = self.calculate_speed(steering_angle)
            self.drive(steering_angle, speed)

            ####################################################################################################################################################################################
            # Visualization stuff
            # Create a copy for visualization and places mask debug image in the top-right corner
            debug_img = image.copy()
            mask_debug_small = cv2.resize(mask_debug, (width//3, height//4))
            debug_img[10:10+mask_debug_small.shape[0], width-10-mask_debug_small.shape[1]:width-10] = mask_debug_small
            
            # Visualize results
            if left_line is not None:
                x1, y1, x2, y2 = left_line
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

            # Draw the center line
            if center_line is not None:
                x1, y1, x2, y2 = center_line
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for center line
                cv2.putText(debug_img, "Center Line", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            ####################################################################################################################################################################################

            
            # Calculate steering angle and speed if goal point is found
            if goal_point is not None:
                # Add debug info about Pure Pursuit
                cv2.putText(debug_img, f"Pure Pursuit: {steering_angle:.2f} rad", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Add status information with steering and speed
                status_text = f"Goal Point: ({goal_point[0]}, {goal_point[1]}) | Steering: {steering_angle:.2f} | Speed: {speed:.1f} m/s"
                cv2.putText(debug_img, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(debug_img, "No intersection found - STOPPED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")


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