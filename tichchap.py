#Tích chập là quá trình đưa Kernel đi qua ảnh để tạo ra ảnh mới.
import cv2
import numpy as np

img = cv2.imread("chainuoc.jpg")

kernel = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
], dtype=np.float32) / 9

result = cv2.filter2D(
    img,
    -1,
    kernel
)

cv2.imshow("Original", img)
cv2.imshow("Blur", result)

cv2.waitKey(0)
