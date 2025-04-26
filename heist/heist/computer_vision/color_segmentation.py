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

def cd_color_segmentation(img, template = None, line = False, merge_bb = False):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the traffic light, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	if line:
		mask = np.zeros(img.shape[:2], np.uint8)
		print(img.shape)
		mask[270:340, :] = 255  # Select the bottom 70 rows
		img = cv2.bitwise_and(img, img, mask=mask)
    
	bounding_box = ((0,0),(0,0))
	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_hue = 1  # Lower HSV threshold
	upper_hue = 30 # Upper HSV threshold

	

	
	if line:
		lower_bound = np.array([lower_hue, 50, 50],  dtype=np.uint8)  # Lower HSV threshold 
		#lower_bound = np.array([1, 120, 140], dtype=np.uint8)  # More lenient on S & V
		#upper_bound = np.array([30, 255, 255], dtype=np.uint8)  # Cover more brightness
	else:
		lower_bound = np.array([lower_hue, 200, 150],  dtype=np.uint8)  # Lower HSV threshold 

	upper_bound = np.array([upper_hue, 255, 255],  dtype=np.uint8)  # Upper HSV threshold
	
	
	mask = cv2.inRange(hsv_image, lower_bound, upper_bound) 
	bounding_boxes = []
	areas = []
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		if cv2.contourArea(contour) > 5:  # Minimum contour area to avoid noise
			x, y, w, h = cv2.boundingRect(contour)
			bounding_box = ( (x, y), (x + w, y + h))
			# cv2.rectangle(img, bounding_box[0], bounding_box[1], (0, 255, 0), 2)  # Draw bounding box
			bounding_boxes.append(bounding_box)
			areas.append(w*h)
	# if merge_bb:
	# 	bounding_boxes = merge_bounding_boxes(bounding_boxes)
	# largest_bbox = max(bounding_boxes, key=lambda box: (box[1][0] - box[0][0]) * (box[1][1] - box[0][1]))
	#cv2.rectangle(img, bounding_boxes[np.argmax(areas)][0], bounding_boxes[np.argmax(areas)][1], (0, 255, 0), 2)  # Draw bounding box
	#image_print(img)
	if bounding_box == ((0,0),(0,0)):
		return bounding_box
	return bounding_boxes[np.argmax(areas)]#largest_bbox

if __name__ == "__main__":
	# cone_template = "test_images_cone/cone_template.png"
	# template = cv2.imread(cone_template)
	# for i in range(1,20):
	# 	image_path = "test_images_cone/test" +str(i) + ".jpg"

	# 	image = cv2.imread(image_path)
	# 	print(cd_color_segmentation(np.array(image), template = None, line = False))
	# for i in range(1,5):
	# 	image_path = "test_images_cone_real/test" +str(i) + ".png"

	# 	image = cv2.imread(image_path)
	# 	cd_color_segmentation(np.array(image))
	image_path = "test_images_line/line1.png"
	image = cv2.imread(image_path)
	cd_color_segmentation(np.array(image), line = False)
