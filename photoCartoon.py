import cv2 as cv
import numpy as np

# photo to cartoon using ChatGPS
img = cv.imread('flowers.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 3)
edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)
color = cv.bilateralFilter(img, 9, 300, 300)
cartoon = cv.bitwise_and(color, color, mask=edges)

cv.imshow("Cartoon", cartoon)
cv.waitKey(0)
cv.destroyAllWindows()