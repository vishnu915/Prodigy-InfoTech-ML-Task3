import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Path to the dataset (after unzipping kagglecatsanddogs_5340.zip)
DATADIR = "PetImages"
CATEGORIES = ["Cat", "Dog"]
IMG_SIZE = 100  # Resize all images to 100x100

data = []
labels = []

# Load and preprocess images
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    label = CATEGORIES.index(category)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_array.flatten())  # Flatten the 2D image to 1D
            labels.append(label)
        except Exception as e:
            # Ignore corrupt images
            continue

print("✅ Total samples loaded:", len(data))

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')  # You can try 'rbf' or 'poly' too
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎯 SVM Model Accuracy:", round(accuracy * 100, 2), "%")
