#NOTE: make simpler, dont need to merge bounding boxes
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################
def merge_bounding_boxes(bboxes, threshold=10):
    """
    Merge bounding boxes that are close to each other.

    Parameters:
        bboxes (list of tuples): List of bounding boxes in the form ((x1, y1), (x2, y2)).
        threshold (int): Minimum distance between bounding boxes to consider them separate.

    Returns:
        merged_bboxes (list of tuples): List of merged bounding boxes.
    """
    merged_bboxes = []
    while bboxes:
        # Start with the first bounding box
        x1, y1 = bboxes[0][0]
        x2, y2 = bboxes[0][1]
        bboxes.pop(0)
        merged = [(x1, y1, x2, y2)]

        # Check for overlapping or nearby bounding boxes
        i = 0
        while i < len(bboxes):
            (bx1, by1),(bx2, by2)= bboxes[i]

            if not (bx2 < x1 - threshold or bx1 > x2 + threshold or  # check if boxes are overlapping or within the threshold
                    by2 < y1 - threshold or by1 > y2 + threshold):
                x1, y1, x2, y2 = min(x1, bx1), min(y1, by1), max(x2, bx2), max(y2, by2) # if so expand the current bounding box
                merged.append((bx1, by1, bx2, by2))
                bboxes.pop(i)  # remove merged mox
            else:
                i += 1  # move to the next box

        merged_bboxes.append(((x1, y1), (x2, y2)))

    return merged_bboxes
def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the traffic light, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	
    
	bounding_box = ((0,0),(0,0))
	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_hue, upper_hue = 0, 15
	lower_saturation, upper_saturation = 200, 230
	lower_value, upper_value = 150, 240

	# Create lower and upper bounds
	lower_bound = np.array([lower_hue, lower_saturation, lower_value], dtype=np.uint8)
	upper_bound = np.array([upper_hue, upper_saturation, upper_value], dtype=np.uint8)

	# Create the mask
	mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
	cv2.imshow("HSV Mask", mask)
	cv2.waitKey(0)  # Waits indefinitely until a key is pressed
	cv2.destroyAllWindows()  # Closes all OpenCV windows
	# Process contours
	best_bounding_box = bounding_box
	biggest_area = 0
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		area = cv2.contourArea(contour)
		
		if area > 150:
			x, y, w, h = cv2.boundingRect(contour)
			aspect_ratio = w / h if h != 0 else 0

			# Check if aspect ratio is close to 1 (square-like)
			if 0.75 <= aspect_ratio <= 1.25:				
				bounding_box = ((x, y), (x + w, y + h))
				if w * h > biggest_area:
					biggest_area = w * h
					# print(biggest_area)

					best_bounding_box = bounding_box
	if biggest_area == 0:
		return ((0,0),(0,0))
	cv2.rectangle(image, best_bounding_box[0], best_bounding_box[1], (0, 255, 0), 2)
	return best_bounding_box


# 	lower_hue, upper_hue = 0,20  
# 	lower_brightness, upper_brightness = 130, 255 
	
# 	#We need to dark reds (nonsaturaed) and bright reds (saturated) due to glare effect
# 	lower_bound1 = np.array([lower_hue, 180, lower_brightness], dtype=np.uint8) 
# 	upper_bound1 = np.array([upper_hue, 225, upper_brightness], dtype=np.uint8)
# 	mask1 = cv2.inRange(hsv_image, lower_bound1, upper_bound1)

# 	lower_bound2 = np.array([lower_hue, 0, lower_brightness], dtype=np.uint8)
# 	upper_bound2 = np.array([upper_hue, 10, upper_brightness], dtype=np.uint8)
# 	mask2 = cv2.inRange(hsv_image, lower_bound2, upper_bound2)

# # Combine both masks
# 	mask = cv2.bitwise_or(mask1, mask2)
# 	best_bounding_box = bounding_box
# 	biggest_area = 0
# 	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if __name__ == "__main__":
	# for i in range(1, 6):
	# 	image_path = f"test_images/stopdistance/light{i}.png"
	# 	image = cv2.imread(image_path)
		
	# 	if image is None:
	# 		print(f"Failed to load {image_path}")
	# 		continue

	# 	bounding_box = cd_color_segmentation(np.array(image))
		
	# 	# Only draw the bounding box if it's valid
	# 	# if bounding_box != ((0, 0), (0, 0)):
	# 	# 	cv2.rectangle(image, bounding_box[0], bounding_box[1], (0, 255, 0), 2)
	# 	print(bounding_box)
	# 	image_print(np.array(image))
	image_path = "test_images/realredlight_full.png"
	image = cv2.imread(image_path)
	bounding_box = cd_color_segmentation(np.array(image))
	cv2.rectangle(image, bounding_box[0], bounding_box[1], (0, 255, 0), 2)
	image_print(np.array(image))
