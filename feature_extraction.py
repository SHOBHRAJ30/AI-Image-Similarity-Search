import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load pre-trained model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img, verbose=0)
    return features.flatten()

image_folder = "dataset"
features_list = []
image_names = []

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    if img_path.lower().endswith((".jpg", ".png", ".jpeg")):
        features = extract_features(img_path)
        features_list.append(features)
        image_names.append(img_name)

np.save("features.npy", np.array(features_list))
np.save("images.npy", np.array(image_names))

print("✅ Features extracted successfully")
print("Total images:", len(image_names))
