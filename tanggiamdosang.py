import cv2 

img = cv2.imread("chainuoc.jpg") 

bright = cv2.add(img,50) # add độ sáng(giảm độ sáng thì dùng dark = cv2.subtract(img,50))

cv2.imshow("Original",img) 
cv2.imshow("Bright",bright) 

cv2.waitKey(0) 