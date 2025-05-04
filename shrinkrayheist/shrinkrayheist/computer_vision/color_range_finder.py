import cv2
import numpy as np
import matplotlib.pyplot as plt
def crop_50x50(image, corner: str, x: int, y: int):
    if corner == "bottom-right":
        x1 = x - 30
        y1 = y - 30
    elif corner == "bottom-left":
        x1 = x
        y1 = y - 30
    else:
        raise ValueError("corner must be either 'bottom-right' or 'bottom-left'")
    
    # Ensure bounds are valid
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = x1 + 30
    y2 = y1 + 30

    # Crop
    return image[y1:y2, x1:x2]
# Load and convert image
image = cv2.imread("test_images/racecarredlight.png")

# Crop the image (store result in 'cropped', not 'iamge')
image = crop_50x50(image, corner="bottom-right", x=480, y=100)
cv2.imshow("Top Middle-Right 50x50", image)
cv2.waitKey(0)
height, width, _ = image.shape

# Coordinates for a 50x50 square in the top-middle-right


# Crop the region

cv2.destroyAllWindows()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)


# Compute the min, max, 5th and 95th percentiles for each channel
h_min, h_max = h.min(), h.max()
h_5th, h_95th = np.percentile(h, [5, 95])

s_min, s_max = s.min(), s.max()
s_5th, s_95th = np.percentile(s, [5, 95])

v_min, v_max = v.min(), v.max()
v_5th, v_95th = np.percentile(v, [5, 95])

print("Hue Min-Max:           ", h_min, "-", h_max)
print("Hue 5th-95th Percentiles:", h_5th, "-", h_95th)

print("Saturation Min-Max:    ", s_min, "-", s_max)
print("Saturation 5th-95th Percentiles:", s_5th, "-", s_95th)

print("Value Min-Max:         ", v_min, "-", v_max)
print("Value 5th-95th Percentiles:   ", v_5th, "-", v_95th)

# Helper to plot and annotate histograms
def plot_hist(channel, color, title, bins, range_, min_val, max_val, perc_5th, perc_95th):
    plt.hist(channel.ravel(), bins=bins, range=range_, color=color, alpha=0.7)
    plt.axvline(min_val, color='k', linestyle='solid', linewidth=1, label=f"Min: {min_val}")
    plt.axvline(max_val, color='k', linestyle='dotted', linewidth=1, label=f"Max: {max_val}")
    plt.axvline(perc_5th, color='b', linestyle='dashed', linewidth=1, label=f"5th Percentile: {perc_5th}")
    plt.axvline(perc_95th, color='b', linestyle='dotted', linewidth=1, label=f"95th Percentile: {perc_95th}")
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Pixel Count')
    plt.legend()

# Plot all three histograms
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plot_hist(h, 'r', 'Hue Histogram', 180, [0, 180], h_min, h_max, h_5th, h_95th)

plt.subplot(1, 3, 2)
plot_hist(s, 'g', 'Saturation Histogram', 256, [0, 255], s_min, s_max, s_5th, s_95th)

plt.subplot(1, 3, 3)
plot_hist(v, 'b', 'Value Histogram', 256, [0, 255], v_min, v_max, v_5th, v_95th)

plt.tight_layout()
plt.show()
