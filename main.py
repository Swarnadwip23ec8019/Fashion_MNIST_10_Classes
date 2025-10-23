import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Function: Model Prediction
# ---------------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')

    # Load and preprocess the image
    image = Image.open(test_image).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))               # Resize to match training size
    input_arr = np.array(image)

    # Normalize and reshape to (1, 28, 28, 1)
    input_arr = input_arr / 255.0
    input_arr = input_arr.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="Fashion MNIST Classifier", page_icon="👗", layout="centered")

st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.selectbox("Choose Page", ["Home", "About", "Fashion MNIST Recognition"])

# ---------------------------
# Home Page
# ---------------------------
if app_mode == "Home":
    st.title("👗 Fashion MNIST Clothing Recognition App")
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
    ### Welcome!
    Upload an image of a clothing item (28×28 grayscale preferred),  
    and the model will predict which category it belongs to!
    """)

# ---------------------------
# About Page
# ---------------------------
elif app_mode == "About":
    st.title("ℹ️ About")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** trained on the  
    **Fashion MNIST dataset**, which contains 70,000 grayscale images  
    of 10 clothing categories.

    **Model Info:**
    - Input size: 28 × 28 pixels  
    - Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot  
    - Framework: TensorFlow / Keras  
    """)

# ---------------------------
# Clothing Recognition Page
# ---------------------------
elif app_mode == "Fashion MNIST Recognition":
    st.title("👕 Fashion MNIST Recognition")
    test_image = st.file_uploader("📤 Upload a clothing image (PNG/JPG):", type=["jpg", "jpeg", "png"])
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Model is predicting..."):
            result_index = model_prediction(test_image)

            # Label map (matches your training dictionary)
            class_names = [
                    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
            ]

            predicted_label = class_names[result_index]
            match predicted_label:
                case 'T-shirt/top':
                    s = '👕'
                case 'Trouser':
                    s = '👖'
                case 'Pullover':
                    s = '🧥'
                case 'Dress':
                    s = '👚'
                case 'Coat':
                    s = '🤵🏻'
                case 'Sandal':
                    s = '🩴'
                case 'Shirt':
                    s = '👕'
                case 'Sneaker':
                    s = '👟'
                case 'Bag':
                    s = '👜'
                case 'Ankle Boot':
                    s = '👢'

            st.success(f"✅ The model predicts this is a **{predicted_label}** "+ s)
