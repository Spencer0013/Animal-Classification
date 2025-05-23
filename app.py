import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tf_keras as keras
import tensorflow_hub as hub 

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        'model.keras',
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    return model

# Load the model
model = load_model()

# Define class names as per the notebook
class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# Streamlit app setup
st.title('Animal Image Classification')
st.write('Upload an image to classify the animal.')

# File uploader for image input
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open and convert the image to RGB
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    
    # Preprocess the image: resize to 224x224 and normalize
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    # Get the predicted label
    predicted_label = class_names[predicted_class]
    
    # Display the prediction
    st.write(f'Prediction: {predicted_label} (Confidence: {confidence * 100:.2f}%)')

