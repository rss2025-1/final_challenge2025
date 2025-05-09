#!/usr/bin/env python3
"""
lane_detector.py

A clean and concise module for lane detection.
API:
  detect_lanes(rgb_image) -> (output_image, x_position, theta)

When run as a script, the module processes a video (“output_video.mp4”) and displays
the lane detection results.
"""

import cv2
import numpy as np
import math

# ------------------- CONSTANTS -------------------
GAUSSIAN_KERNEL = (5, 5)
THRESHOLD_VALUE = 180
HOUGH_RHO = 1
HOUGH_THETA = math.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 30
HOUGH_MAX_LINE_GAP = 20
# For keystone warp: the warped ROI uses these destination ratios on the top edge.
WARP_TOP_LEFT_RATIO = 0.0
WARP_TOP_RIGHT_RATIO = 1.0
WARP_BOTTOM_LEFT_MULT = 0.45
WARP_BOTTOM_RIGHT_MULT = 0.55

# ------------------- HELPER FUNCTIONS -------------------
def preprocess_image(rgb_image):
    """Convert the RGB image to grayscale, blur it, and threshold to produce a binary mask."""
    # Use cvtColor with COLOR_RGB2GRAY because input is RGB.
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    _, thresh = cv2.threshold(blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    return thresh

def get_roi(binary_image):
    """Return the bottom half of the image."""
    h = binary_image.shape[0]
    return binary_image[h // 2 :, :]

def keystone_warp(image):
    """
    Applies a keystone (perspective) warp on the supplied image.
    Returns the warped image along with the source and destination points
    used in the warp.
    """
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0],
                             [w, 0],
                             [w, h],
                             [0, h]])
    dest_points = np.float32([[0, 0],
                              [w, 0],
                              [w * WARP_BOTTOM_RIGHT_MULT, h],
                              [w * WARP_BOTTOM_LEFT_MULT, h]])
    M = cv2.getPerspectiveTransform(src_points, dest_points)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped, src_points, dest_points

def compute_x_at_y(line, y_val, epsilon=1e-6):
    """
    Given a line (tuple of two endpoints), compute the x-value corresponding to y_val.
    If the line is nearly horizontal, return the average x.
    """
    (x1, y1), (x2, y2) = line
    dy = y2 - y1
    if abs(dy) < epsilon:
        return int((x1 + x2) / 2)
    t = (y_val - y1) / dy
    x = x1 + t * (x2 - x1)
    return int(x)

def averageCenterLine(left_line, right_line, height):
    """
    Compute an averaged centerline from left and right lines in warped coordinates.
    The function computes the x-values at the top (y=0) and bottom (y=height) for each line,
    then averages the corresponding x-values.
    """
    top_y = 0
    bot_y = height
    left_top = compute_x_at_y(left_line, top_y)
    left_bot = compute_x_at_y(left_line, bot_y)
    right_top = compute_x_at_y(right_line, top_y)
    right_bot = compute_x_at_y(right_line, bot_y)
    
    center_top = (left_top + right_top) / 2.0
    center_bot = (left_bot + right_bot) / 2.0
    return np.array([[center_top, top_y],
                     [center_bot, bot_y]], dtype=np.float32)

def unwarp_centerline(center_line_warped, image):
    """
    Re-project the centerline, computed in the warped (ROI) coordinate system,
    back into the coordinate system of the original image.
    roi_offset: vertical offset to add (because the ROI was the bottom half).
    """
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0],
                             [w, 0],
                             [w, h],
                             [0, h]])
    dest_points = np.float32([[0, 0],
                              [w, 0],
                              [w * WARP_BOTTOM_RIGHT_MULT, h],
                              [w * WARP_BOTTOM_LEFT_MULT, h]])
    roi_offset = h // 2
    inverse_M = cv2.getPerspectiveTransform(dest_points, src_points)
    pts = center_line_warped.reshape(-1, 1, 2)
    unwarped = cv2.perspectiveTransform(pts, inverse_M)
    unwarped_line = unwarped.reshape(-1, 2)
    # Add back the vertical offset (the ROI was taken from bottom half)
    unwarped_line[:, 1] += roi_offset
    return unwarped_line

def detect_lines(warped_binary):
    """
    Run HoughLinesP on the warped binary image.
    Separate detected lines into left and right groups (by the midpoint x value).
    Return the longest line candidate from each group.
    If a candidate is missing, None is returned in its place.
    """
    lines = cv2.HoughLinesP(warped_binary,
                            rho=HOUGH_RHO,
                            theta=HOUGH_THETA,
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LENGTH,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    if lines is None:
        return None, None

    left_lines = []
    right_lines = []
    h, w = warped_binary.shape
    mid_x = w / 2.0

    # Process each line from Hough transform
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Skip nearly horizontal lines (angle < 45°)
        if abs(math.degrees(math.atan2((y2 - y1), (x2 - x1)))) < 60:
            continue
        # Separate left and right based on the midpoint
        line_mid = (x1 + x2) / 2.0
        if line_mid < mid_x:
            left_lines.append(((x1, y1), (x2, y2)))
        else:
            right_lines.append(((x1, y1), (x2, y2)))

    # From each group, choose the longest line as candidate.
    def line_length(line):
        (x1, y1), (x2, y2) = line
        return math.hypot(x2 - x1, y2 - y1)

    left_candidate = max(left_lines, key=line_length) if left_lines else None
    right_candidate = max(right_lines, key=line_length) if right_lines else None

    return left_candidate, right_candidate

def overlay_centerline(orig_image, line_pts, color=(255, 0, 0), thickness=5):
    """Draw a line defined by two points on orig_image."""
    pt1 = tuple(np.int32(line_pts[0]))
    pt2 = tuple(np.int32(line_pts[1]))
    cv2.line(orig_image, pt1, pt2, color, thickness)
    return orig_image

def compute_line_angle(line_pts):
    """
    Given two endpoints [ [x1, y1], [x2, y2] ], compute the centerline angle (theta)
    in degrees relative to the horizontal.
    """
    (x1, y1), (x2, y2) = line_pts
    dx = x2 - x1
    dy = y2 - y1
    # Use arctan2: note that y increases downward.
    theta = math.degrees(math.atan2(dy, dx))
    return theta

# ------------------- MAIN API -------------------
def detect_lanes(rgb_image):
    """
    detect_lanes(rgb_image) -> output_image, x_position, theta

    Process the provided RGB image to detect road lanes.
    The function returns:
      - output_image: The original image with the computed centerline overlaid.
      - x_position: The x coordinate (in original image) at the bottom end of the centerline.
      - theta: The angle (in degrees) relative to horizontal of the centerline.
    If lanes cannot be detected, None is returned for x_position and theta.
    """
    # Preprocess and get binary mask.
    binary = preprocess_image(rgb_image)
    orig_h, orig_w = binary.shape
    roi = get_roi(binary)  # bottom half of the image

    # Apply keystone warp on the ROI.
    warped_roi, warp_src_pts, warp_dest_pts = keystone_warp(roi)
    # For Hough transform, work on the warped (binary) image.
    # (No need to convert color for detection.)
    
    # Detect left and right line candidates.
    left_line, right_line = detect_lines(warped_roi)
    # If we cannot obtain both lines, return the original image.
    output_image = rgb_image.copy()
    if left_line is None or right_line is None:
        return output_image, None, None

    center_line_warped = averageCenterLine(left_line, right_line, orig_h)
    # Unwarp the centerline back into original image coordinates.
    center_line_unwarped = unwarp_centerline(center_line_warped, rgb_image)
    # Overlay the centerline on the original image.
    output_image = overlay_centerline(output_image, center_line_unwarped)

    # Compute x-position: use the bottom endpoint of the centerline.
    x_position = center_line_unwarped[1, 0]
    theta = compute_line_angle(center_line_unwarped)

    return output_image, x_position, theta

# ------------------- MAIN (VIDEO PROCESSING DEMO) -------------------
def main():
    """Process video frames from output_video.mp4 and display lane detection."""
    video_capture = cv2.VideoCapture("output_video.mp4")
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    print("Starting video processing... Press 'q' to exit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # OpenCV reads video frames as BGR. Convert to RGB for our API.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_img, x_pos, theta = detect_lanes(frame_rgb)
        
        # For display, convert output back to BGR.
        out_display = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Lane Detection", out_display)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Video processing ended.")

if __name__ == "__main__":
    main()