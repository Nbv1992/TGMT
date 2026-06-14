import cv2
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================
# Đọc dữ liệu ảnh
# ==========================

data = []
labels = []

classes = ["cat", "dog"]

for label, folder in enumerate(classes):

    folder_path = os.path.join("dataset", folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize ảnh
        img = cv2.resize(img, (64, 64))

        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Chuyển thành vector
        feature = gray.flatten()

        data.append(feature)
        labels.append(label)

# ==========================
# Chuyển sang NumPy
# ==========================

X = np.array(data, dtype=np.float32)
y = np.array(labels)

# Chuẩn hóa dữ liệu
X = X / 255.0

# ==========================
# Chia tập train/test
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ==========================
# Tạo Shallow Neural Network
# ==========================

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    max_iter=500,
    random_state=42
)

# ==========================
# Huấn luyện
# ==========================

model.fit(X_train, y_train)

# ==========================
# Dự đoán
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Đánh giá
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Dự đoán ảnh mới
# ==========================

test_image = "test.jpg"

img = cv2.imread(test_image)

if img is not None:

    img_resize = cv2.resize(img, (64, 64))

    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    feature = gray.flatten().astype(np.float32)

    feature = feature / 255.0

    feature = feature.reshape(1, -1)

    result = model.predict(feature)

    label = classes[result[0]]

    print("\nKết quả dự đoán:", label)

    cv2.putText(
        img,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Khong tim thay anh test.jpg")