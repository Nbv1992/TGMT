import cv2

img = cv2.imread("chainuoc.jpg")

negative = 255 - img

cv2.imshow("Original", img)
cv2.imshow("Negative", negative)

cv2.waitKey(0)
