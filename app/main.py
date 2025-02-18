import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import clean_image, get_prediction, make_results

working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = f"{working_dir}/trained_model/plant_disease_model.h5"

# Define model architecture (if using state_dict)
class PlantDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes=14):  # Adjust based on your model
        super(PlantDiseaseModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(32 * 224 * 224, num_classes)  # Adjust based on architecture

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

torch.serialization.add_safe_globals([PlantDiseaseModel])

# Load Model
@st.cache_resource
def load_model():
    model_path = f"{working_dir}/trained_model/plant-disease-model-full.pth"

    try:
        # ✅ Option 1: Load full model
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    except:
        # ✅ Option 2: Load weights into defined architecture
        model = PlantDiseaseModel(num_classes=14)  # Adjust if needed
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))

    model.eval()
    return model

model = load_model()

# Define Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Streamlit UI
st.title("Plant Disease Detection Tool")
st.write("Upload an image of a plant leaf to predict if it's healthy or diseased.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        image_tensor = preprocess_image(image)

        # Run prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Get class labels (adjust based on your dataset)
        class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                        'Apple___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy']
        
        predicted_label = class_labels[predicted_class]
        confidence = probabilities[0, predicted_class].item() * 100

        st.success(f"Prediction: **{predicted_label}** with {confidence:.2f}% confidence.")







# # Load pre-trained model
# model = tf.keras.models.load_model(model_path)

# # Load class names
# class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# index_class = {int(k): v for k, v in class_indices.items()}
# # Extract Healthy Class Names (Any label containing 'healthy')
# HEALTHY_CLASSES = {v for v in class_indices.values() if "healthy" in v.lower()}

# # Load and Preprocess the Image 
# def load_and_preprocess_image(image_path, target_size=(224, 224)):
#     img = Image.open(image_path)
#     img = img.resize(target_size)

#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array.astype('float32') / 255.

#     return img_array

# # Predict Image Class (Healthy or Unhealthy)
# def predict_health_status(model, image_path):
#     preprocessed_img = load_and_preprocess_image(image_path)
#     predictions = model.predict(preprocessed_img)
    
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class_name = index_class[predicted_class_index]

#     # Determine health status
#     if predicted_class_name in HEALTHY_CLASSES:
#         status = "is Healthy "
#     else:
#         status = "is Unhealthy "

#     return {
#         "status": status,
#         "prediction": predicted_class_name
#     }

# # Function to Predict the Class of an Image
# def predict_image_class(model, image_path, class_indices):
#     preprocessed_img = load_and_preprocess_image(image_path)
#     predictions = model.predict(preprocessed_img)
#     predicted_class_index = np.argmax(predictions, axis=1)[0]
#     predicted_class_name = class_indices[str(predicted_class_index)]
    
#     result = make_results(predictions, predicted_class_index)

#     return result
#     # return predicted_class_name


# # Streamlit App
# st.title('Plant Disease Detection Tool')

# uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     image = Image.open(uploaded_image)
#     col1, col2 = st.columns(2)

#     with col1:
#         resized_img = image.resize((150, 150))
#         st.image(resized_img)

#     with col2:
#         if st.button('Check'):
#             # prediction = predict_image_class(model, uploaded_image, class_indices)
#             result = predict_health_status(model, uploaded_image)
#             st.success(f"The plant {result['status']} with a {result['prediction']} prediction.")
#             # st.success(f'Model Prediction: {str(prediction)}')
#             # st.write(f"The plant {prediction['status']} with {prediction['prediction']} prediction.")
