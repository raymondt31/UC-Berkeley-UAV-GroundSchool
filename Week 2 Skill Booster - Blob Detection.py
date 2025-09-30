import cv2
import numpy as np

image = cv2.imread("polka_dots_2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(image, (5, 5), 0)

params = cv2.SimpleBlobDetector_Params()

#Set Thresholds
params.minThreshold = 0
params.maxThreshold = 255
params.thresholdStep = 3

#Enable Filters
params.filterByArea = True
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

params.minArea = 100
params.maxArea = 10000
params.minCircularity = 0.5
params.minConvexity = 0.4
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(blur)

blobs = cv2.drawKeypoints(blur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#cv2.imshow("Blur",blur)
cv2.imshow("Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

#for kp in keypoints:
    #print(kp.size, kp.pt)