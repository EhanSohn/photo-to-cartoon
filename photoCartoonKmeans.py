import cv2 as cv
import numpy as np

# photo to cartoon using Kmeans
img = cv.imread('flowers.png')
imgTrans = img.reshape((-1, 3)).astype(np.float32)
criteria = (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
compactness, label, center = cv.kmeans(imgTrans, 15, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
cartoonKmeans = center[label.flatten()]
cartoonKmeans = cartoonKmeans.reshape((img.shape))

cv.imshow("Cartoon", cartoonKmeans)
cv.waitKey(0)
cv.destroyAllWindows()