import argparse
import cv2
import numpy as np


def classify_animal(image):
    """Phân loại cơ bản: chó, mèo hoặc lợn bằng màu sắc và tông xám."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(image)
    total_pixels = image.shape[0] * image.shape[1]

    pink_mask = ((h < 10) | (h > 160)) & (s > 50) & (v > 120)
    pink_ratio = np.count_nonzero(pink_mask) / total_pixels

    gray_mask = (np.abs(b - g) < 20) & (np.abs(g - r) < 20) & (v > 50) & (v < 200)
    gray_ratio = np.count_nonzero(gray_mask) / total_pixels

    brown_mask = (h >= 5) & (h <= 25) & (s > 50) & (v > 50)
    brown_ratio = np.count_nonzero(brown_mask) / total_pixels

    if pink_ratio > 0.07 and gray_ratio < 0.30:
        return "Pig"
    if gray_ratio > 0.14 and pink_ratio < 0.20:
        return "Cat"
    if brown_ratio > 0.08:
        return "Dog"

    if gray_ratio > 0.12:
        return "Cat"
    return "Dog"


def label_image(image, label):
    output = image.copy()
    cv2.putText(output, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return output


def main():
    parser = argparse.ArgumentParser(description="Phân loại đơn giản 3 ảnh thành chó, mèo, lợn.")
    parser.add_argument("images", nargs=3, help="Ba đường dẫn ảnh cần phân loại")
    args = parser.parse_args()

    for idx, image_path in enumerate(args.images, 1):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không đọc được ảnh: {image_path}")
            continue

        label = classify_animal(image)
        output = label_image(image, label)
        result_path = f"result_{idx}_{label}.png"
        cv2.imwrite(result_path, output)
        print(f"Ảnh {image_path} -> {label}, lưu: {result_path}")

        window_name = f"Image {idx}: {label}"
        cv2.imshow(window_name, output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
