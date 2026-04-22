from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os   # 👈 add this

# load model
model = tf.keras.models.load_model("model.h5")

# class names
class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# =========================
# USER INPUT
# =========================
img_path = input("Enter image name (e.g. inclusion_2.jpg): ")

# 🔴 check if file exists
if not os.path.exists(img_path):
    print("❌ Image not found! Check name or location.")
    exit()

# load image
img = image.load_img(img_path, target_size=(224, 224))

# convert to array
img_array = image.img_to_array(img)

# expand dims
img_array = np.expand_dims(img_array, axis=0)

# normalize
img_array = img_array / 255.0

# prediction
prediction = model.predict(img_array)

# result
predicted_class = class_names[np.argmax(prediction)]

# print result
print("✅ Prediction:", predicted_class)

# show image
plt.imshow(img)
plt.title("Prediction: " + predicted_class)
plt.axis('off')
plt.show()