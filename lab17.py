import cv2
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

        # Chuyển ảnh xám
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
# Chia train/test
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ==========================
# Xây dựng MLP
# ==========================

model = tf.keras.Sequential([

    tf.keras.layers.Dense(
        256,
        activation='relu',
        input_shape=(4096,)
    ),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        64,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        2,
        activation='softmax'
    )
])

# ==========================
# Compile model
# ==========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Huấn luyện
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# ==========================
# Đánh giá
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nAccuracy:", accuracy)

# ==========================
# Dự đoán tập test
# ==========================

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

print(
    "Test Accuracy:",
    accuracy_score(y_test, y_pred)
)

# ==========================
# Dự đoán ảnh mới
# ==========================

img = cv2.imread("test.jpg")

if img is not None:

    img_resize = cv2.resize(
        img,
        (64, 64)
    )

    gray = cv2.cvtColor(
        img_resize,
        cv2.COLOR_BGR2GRAY
    )

    feature = gray.flatten()

    feature = feature.astype(np.float32)

    feature = feature / 255.0

    feature = feature.reshape(1, -1)

    prediction = model.predict(feature)

    result = np.argmax(prediction)

    label = classes[result]

    print("\nPrediction:", label)

    cv2.putText(
        img,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Prediction",
        img
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Khong tim thay test.jpg")

# ==========================
# Lưu model
# ==========================

model.save("animal_classifier.keras")

print("\nDa luu model!")