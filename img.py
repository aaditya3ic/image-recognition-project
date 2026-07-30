# ==============================================================================
# PROJECT 2: IMAGE RECOGNITION (Computer Vision Pipeline in Pure Python)
# ==============================================================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array
import numpy as np
import urllib.request
import os
import time
from PIL import Image

# ------------------------------------------------------------------------------
# STEP 1: Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Image Recognition AI", page_icon="📸", layout="wide")
st.title("📸 Image Recognition AI (Transfer Learning)")
st.write("A Computer Vision model using MobileNetV2 to classify images into 1,000 different categories.")

# ------------------------------------------------------------------------------
# STEP 2: Model Loading (Cached for Web Speed)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Downloads ~14MB of pretrained weights on the first run.
    model = MobileNetV2(weights="imagenet")
    return model

with st.spinner("Loading pretrained MobileNetV2 model... (This may take a moment on shared Wi-Fi)"):
    image_model = load_model()

# ------------------------------------------------------------------------------
# STEP 3: Image Upload & Fallback Logic
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.write(f"**TensorFlow Version:** {tf.__version__}")
st.sidebar.write("---")

st.subheader("1. Provide an Image")
st.write("Upload a photo of a cat or kitten, handcrafted items like a crochet bag, or everyday objects to see how the model reacts!")

uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Fallback image logic if the user hasn't uploaded anything yet
FALLBACK_IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
FALLBACK_IMAGE_PATH = "sample_image.jpg"

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.success("Image uploaded successfully!")
else:
    # Use the fallback image
    if not os.path.exists(FALLBACK_IMAGE_PATH):
        urllib.request.urlretrieve(FALLBACK_IMAGE_URL, FALLBACK_IMAGE_PATH)
    image = Image.open(FALLBACK_IMAGE_PATH)
    st.info("ℹ️ No image uploaded yet. Using standard fallback sample image.")

# Display the image in the UI
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Input Image", use_column_width=True)

# ------------------------------------------------------------------------------
# STEP 4: Preprocessing & Prediction
# ------------------------------------------------------------------------------
with col2:
    st.subheader("2. AI Analysis")
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Processing image grid..."):
            
            # Convert image to RGB (in case of Grayscale or PNG with transparency)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Resize exactly to 224x224 as required by MobileNetV2
            resized_image = image.resize((224, 224))
            
            # Convert to NumPy array
            image_array = img_to_array(resized_image)
            
            # Expand dimensions to create a batch of 1: shape becomes (1, 224, 224, 3)
            image_batch = np.expand_dims(image_array, axis=0)
            
            # Preprocess the input (scales pixel values for this specific model)
            processed_image = preprocess_input(image_batch)
            
            # Make the prediction and measure time
            start_time = time.time()
            predictions = image_model.predict(processed_image, verbose=0)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Decode the top 5 predictions
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            # Display metrics
            st.write(f"⏱️ **Prediction took:** {elapsed_ms:.1f} ms")
            st.write(f"📐 **Batch Shape:** `{image_batch.shape}`")
            
            st.markdown("### Top 5 Predictions:")
            for i, (imagenet_id, label, probability) in enumerate(decoded_predictions):
                # Clean up the label text
                clean_label = label.replace("_", " ").title()
                prob_pct = probability * 100
                
                # Display as a progress bar for visual impact
                st.write(f"**{i+1}. {clean_label}** ({prob_pct:.2f}%)")
                st.progress(float(probability))

# ------------------------------------------------------------------------------
# STEP 5: Educational Resources (For the Presentation)
# ------------------------------------------------------------------------------
st.markdown("---")
st.header("📚 Presentation Talking Points")

with st.expander("🤔 What Actually Happened Here?"):
    st.write("""
    The model did **not** "see" the image the way a human does. It processed a grid of numbers through many mathematical layers, each one transforming those numbers slightly, until the final layer produced a probability for each of the 1,000 categories it was trained to recognize.
    
    This also means the model can be **confidently wrong** — a high percentage score reflects how strongly the pattern matched something it learned during training, not a guarantee of correctness.
    """)

with st.expander("🎯 Student Challenge"):
    st.write("Discuss as a class:")
    st.markdown("""
    1. What happens with a blurry image?
    2. What happens with multiple objects in the frame?
    3. Does the highest confidence score guarantee the prediction is correct?
    
    *This introduces an important idea: AI predictions are probabilistic, and probability is not the same as certainty.*
    """)

with st.expander("💼 Interview Corner"):
    st.markdown("""
    1. **What is transfer learning?** Why does it save time and data compared to training from scratch?
    2. **Why 224x224?** Why do we resize every image to a fixed size before feeding it to the network?
    3. **Preprocessing:** What does `preprocess_input` actually do to the pixel values?
    4. **Probabilities:** Why does the model output probabilities across 1,000 classes instead of one single, certain answer?
    5. **Limitations:** Name one real-world situation where a pretrained ImageNet model would perform poorly.
    """)