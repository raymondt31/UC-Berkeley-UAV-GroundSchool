import cv2
import numpy as np
import matplotlib.pyplot as plt

cherry = cv2.imread('cherry.jpg')
cherry_HSV = cv2.cvtColor(cherry, cv2.COLOR_BGR2HSV)
cv2.imshow("Cherry", cherry_HSV)

lower = np.array([1,0,0])
upper = np.array([180, 255, 235])

mask = cv2.inRange(cherry_HSV, lower, upper)
#cv2.imshow('Mask', mask)

final = cv2.bitwise_and(cherry, cherry, mask=mask)
#cv2.imshow('Results', final)

cv2.waitKey(0)
cv2.destroyAllWindows()