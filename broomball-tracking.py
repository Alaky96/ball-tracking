import argparse
import cv2
import imutils
import numpy as np
import pickle

lower = np.array([0, 0, 0])
upper = np.array([0, 0, 0])
min_radius = 0
max_radius = 0
param1 = 1
param2 = 1
dp = 1

def on_trackbar_change(position):
    return

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--video", required=True, help="path to the video file")
args = vars(parser.parse_args())

# grab the video file
vs = cv2.VideoCapture(args["video"])

# Create window
cv2.namedWindow('image')

# Set mouse callback to capture HSV value on click
cv2.createTrackbar("Min H", "image", int(lower[0]), 255, on_trackbar_change)
cv2.createTrackbar("Min S", "image", int(lower[1]), 255, on_trackbar_change)
cv2.createTrackbar("Min V", "image", int(lower[2]), 255, on_trackbar_change)
cv2.createTrackbar("Max H", "image", int(upper[0]), 255, on_trackbar_change)
cv2.createTrackbar("Max S", "image", int(upper[1]), 255, on_trackbar_change)
cv2.createTrackbar("Max V", "image", int(upper[2]), 255, on_trackbar_change)
cv2.createTrackbar("Min Radius", "image", int(min_radius), 500, on_trackbar_change)
cv2.createTrackbar("Max Radius", "image", int(max_radius), 500, on_trackbar_change)
cv2.createTrackbar("Param 1", "image", int(param1), 500, on_trackbar_change)
cv2.createTrackbar("Param 2", "image", int(param2), 500, on_trackbar_change)
cv2.createTrackbar("DP", "image", int(dp), 500, on_trackbar_change)

while True:
    # Read the next frame
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1]

    # Loop
    if frame is None:
        vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    lower[0] = cv2.getTrackbarPos("Min H", "image")
    lower[1] = cv2.getTrackbarPos("Min S", "image")
    lower[2] = cv2.getTrackbarPos("Min V", "image")
    upper[0] = cv2.getTrackbarPos("Max H", "image")
    upper[1] = cv2.getTrackbarPos("Max S", "image")
    upper[2] = cv2.getTrackbarPos("Max V", "image")
    min_radius = cv2.getTrackbarPos("Min Radius", "image")
    max_radius = cv2.getTrackbarPos("Max Radius", "image")
    param1 = cv2.getTrackbarPos("Param 1", "image")
    param2 = cv2.getTrackbarPos("Param 2", "image")
    dp = cv2.getTrackbarPos("DP", "image")


    frame = imutils.resize(frame, width=600)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Look for circle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred_gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # blurred_gray = cv2.erode(blurred_gray, None, iterations=2)
    # blurred_gray = cv2.dilate(blurred_gray, None, iterations=2)
    blurred_gray = gray

    circles = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp, 5, minRadius=min_radius, maxRadius=max_radius, param1=param1, param2=param2)

    if circles is not None:

        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.circle(blurred_gray, (x, y), r, (0, 255, 0), 4)

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("mask", mask)
    cv2.imshow("gray", blurred_gray)

    # Get trackbar positions and set lower/upper bounds



    # Show range mask
    cv2.imshow("image", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()
