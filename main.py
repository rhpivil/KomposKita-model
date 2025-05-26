from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with GitHub Pages URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.saved_model.load("model")

waste_labels = ["Sampah Organik Basah (LAYAK KOMPOS)", "Sampah Organik Kering (LAYAK KOMPOS)", "Sampah Tidak Layak Kompos"]


def preprocess_image(image_bytes):
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    image_tensor = tf.expand_dims(image, 0)
    return image_tensor.numpy().tolist()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    prediction = model(image_tensor)
    predicted_class = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))

    return {
        "predicted_class": predicted_class,
        "label": waste_labels[predicted_class],
        "confidence": confidence,
    }