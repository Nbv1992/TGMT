import cv2

img = cv2.imread("chainuoc.jpg")

result = cv2.convertScaleAbs(  # hàm thay đổi độ sáng (Brightness) và độ tương phản (Contrast) của ảnh
    img,
    alpha = 1.8,
    beta = 20
)
cv2.imshow("Original",img)
cv2.imshow("Result",result)

cv2.waitKey(0)