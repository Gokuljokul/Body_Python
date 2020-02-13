import cv2

# reading the image
image = cv2.imread("human4.jpeg")
edged = cv2.Canny(image, 10, 250)
cv2.imshow("Edges", edged)
cv2.waitKey(0)