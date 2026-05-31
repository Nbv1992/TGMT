import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)

hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
hist = hist / hist.sum()

# cumulative probability
omega = np.cumsum(hist)

# cumulative mean
mu = np.cumsum(hist * np.arange(256))

mu_t = mu[-1]

sigma_b_max = 0
best_threshold = 0

for t in range(256):

    if omega[t] == 0 or omega[t] == 1:
        continue

    sigma_b = (
        (mu_t * omega[t] - mu[t]) ** 2
        / (omega[t] * (1 - omega[t]))
    )

    if sigma_b > sigma_b_max:
        sigma_b_max = sigma_b
        best_threshold = t

print("Threshold tối ưu =", best_threshold)

_, result = cv2.threshold(
    img,
    best_threshold,
    255,
    cv2.THRESH_BINARY
)

cv2.imshow("Original", img)
cv2.imshow("Optimized Otsu", result)

cv2.waitKey(0)
cv2.destroyAllWindows()