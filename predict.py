import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "package_quality_model.h5"
IMG_SIZE = (224, 224)

CLASS_LABELS = {0: "Damaged", 1: "Good"}

print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!\n")


if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]


img = image.load_img(image_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)


prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    label = CLASS_LABELS[1]
    confidence = prediction
else:
    label = CLASS_LABELS[0]
    confidence = 1 - prediction


print("=" * 45)
print(f"Image      : {image_path}")
print(f"Prediction : {label}")
print(f"Confidence : {confidence*100:.2f}%")
print("=" * 45)