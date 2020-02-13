import cv2
import numpy as np
x1, y1, w1, h1 = (0,0,0,0)
points = 0

# load image
img = cv2.imread('human3.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
# threshold to get just the signature
retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# find where the signature is and make a cropped region
points = np.argwhere(thresh_gray==0) # find where the black pixels are
points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
crop = img[y:y+h, x:x+w]
cv2.imshow('save.jpg', crop)
cv2.waitKey(0)