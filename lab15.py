import cv2
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

data = []
labels = []

classes = ["cat", "dog"]

for label, folder in enumerate(classes):

    folder_path = os.path.join("dataset", folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        img = cv2.resize(img, (64,64))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        feature = img.flatten()

        data.append(feature)

        labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

model = Perceptron(
    max_iter=1000,
    eta0=0.01
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy =", accuracy)