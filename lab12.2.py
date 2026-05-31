import cv2

img = cv2.imread("image.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    print("Area =", area)
    print("Perimeter =", perimeter)

    cv2.drawContours(
        img,
        [cnt],
        -1,
        (0, 255, 0),
        2
    )

cv2.imshow("Result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()