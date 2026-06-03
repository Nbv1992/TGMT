import cv2

img = cv2.imread("chainuoc.jpg")

gamma = 0.5

result = 255 * (img / 255) ** gamma

result = result.astype("uint8") #Chuyển kiểu dữ liệu của ma trận result sang kiểu uint8 (số nguyên không dấu 8 bit).

cv2.imshow("Original", img)
cv2.imshow("Gamma", result)

cv2.waitKey(0)