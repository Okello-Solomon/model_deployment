from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()
# Load the pre-trained model
model = tf.keras.models.load_model('cifar_model.keras')

# Define the class names for CIFAR-10
class_names=["airplane","automobile","bird",
             "cat","deer", "dog","frog","horse","ship","truck"]

#preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((32, 32))  # Resize to the input size of the model
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    
    return {"predicted_class": predicted_class}