import cv2
import numpy as np

img = cv2.imread("chainuoc.jpg")

c = 255 / np.log(1 + 255)

result = c * np.log(1 + img)

result = result.astype(np.uint8)

cv2.imshow("Original", img)
cv2.imshow("Log", result)

cv2.waitKey(0)
