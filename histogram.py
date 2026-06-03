import cv2
import matplotlib.pyplot as plt

img = cv2.imread("chainuoc.jpg", 0) #Đọc ảnh dưới dạng ảnh xám (Grayscale).

hist = cv2.calcHist(
    [img],
    [0], #ảnh chỉ có 1 kênh xám duy nhất BGR=012
    None, #Chỉ định vùng nào của ảnh được xử lý và vùng nào bị bỏ qua(mặt nạ lọc vùng quan tâm)
    [256],
    [0, 256]
)

plt.plot(hist)
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Number of Pixels")
plt.show()