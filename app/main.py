import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

from utils import clean_image, get_prediction, make_results

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_model.h5"

# Load pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

index_class = {int(k): v for k, v in class_indices.items()}
# Extract Healthy Class Names (Any label containing 'healthy')
HEALTHY_CLASSES = {v for v in class_indices.values() if "healthy" in v.lower()}

# Load and Preprocess the Image 
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.

    return img_array

# Predict Image Class (Healthy or Unhealthy)
def predict_health_status(model, image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = index_class[predicted_class_index]

    # Determine health status
    if predicted_class_name in HEALTHY_CLASSES:
        status = "is Healthy "
    else:
        status = "is Unhealthy "

    return {
        "status": status,
        "prediction": predicted_class_name
    }

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    result = make_results(predictions, predicted_class_index)

    return result
    # return predicted_class_name


# Streamlit App
st.title('Plant Disease Detection Tool')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Check'):
            # prediction = predict_image_class(model, uploaded_image, class_indices)
            result = predict_health_status(model, uploaded_image)
            # st.success(f'Model Prediction: {str(prediction)}')
            # st.write(f"The plant {prediction['status']} with {prediction['prediction']} prediction.")
