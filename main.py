from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kompos-kita.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.saved_model.load("model")
model_inference = model.signatures["serving_default"]

waste_labels = ["Sampah Organik Basah (LAYAK KOMPOS)", "Sampah Organik Kering (LAYAK KOMPOS)", "Sampah Tidak Layak Kompos"]


def preprocess_image(image_bytes):
    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    image_tensor = tf.expand_dims(image, 0)
    return image_tensor


@app.post("/api/predicts")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_tensor = preprocess_image(image_bytes)
    prediction = model_inference(image_tensor)
    prediction_values = list(prediction.values())[0].numpy()[0]
    predicted_class = int(np.argmax(prediction_values))
    confidence = float(np.max(prediction_values))

    return {
        "predicted_class": predicted_class,
        "label": waste_labels[predicted_class],
        "confidence": confidence,
    }