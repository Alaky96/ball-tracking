# import the necessary packages
import numpy as np
import argparse
import cv2
import time

orange = np.uint8([[[43, 56, 154]]])
hsv_orange = cv2.cvtColor(orange, cv2.COLOR_BGR2HSV)
print(hsv_orange)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())


# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])

output = image.copy()
src = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(src, (0, 100, 100), (10, 255, 255))
# mask = cv2.erode(mask, None, iterations=2)
# mask = cv2.dilate(mask, None, iterations=2)
cv2.imshow("frame", mask)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect circles in the image
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2, 60, param1=100, param2=15, minRadius=0, maxRadius=10)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    # show the output image
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)
