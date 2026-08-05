import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import os
import time
import json

st.set_page_config(page_title="Image Recognition AI", page_icon="📸", layout="wide")
st.title("📸 Image Recognition AI (Transfer Learning)")
st.write("A Computer Vision model using MobileNetV2 (PyTorch) to classify images into 1,000 different categories.")

# Load the 1000 category labels
@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = urllib.request.urlopen(url)
    labels = [line.decode('utf-8').strip() for line in response.readlines()]
    return labels

# Load the PyTorch MobileNetV2 Model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval() # Set model to evaluation mode
    return model

with st.spinner("Loading pretrained MobileNetV2 model... (This may take a moment)"):
    image_model = load_model()
    imagenet_labels = load_labels()

st.sidebar.header("⚙️ Configuration")
st.sidebar.write(f"**PyTorch Version:** {torch.__version__}")
st.sidebar.write("---")

st.subheader("1. Provide an Image")
st.write("Upload a photo to see how the model reacts!")

uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

FALLBACK_IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
FALLBACK_IMAGE_PATH = "sample_image.jpg"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.success("Image uploaded successfully!")
else:
    if not os.path.exists(FALLBACK_IMAGE_PATH):
        urllib.request.urlretrieve(FALLBACK_IMAGE_URL, FALLBACK_IMAGE_PATH)
    image = Image.open(FALLBACK_IMAGE_PATH)
    st.info("ℹ️ No image uploaded yet. Using standard fallback sample image.")

col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Input Image", use_container_width=True)
with col2:
    st.subheader("2. AI Analysis")
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Processing image..."):
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # PyTorch Preprocessing Pipeline
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # Add batch dimension
            
            start_time = time.time()
            with torch.no_grad():
                output = image_model(input_batch)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            st.write(f"⏱️ **Prediction took:** {elapsed_ms:.1f} ms")
            st.write(f"📐 **Batch Shape:** `(1, 3, 224, 224)`")
            
            st.markdown("### Top 5 Predictions:")
            for i in range(5):
                label = imagenet_labels[top5_catid[i]].replace("_", " ").title()
                prob_pct = top5_prob[i].item() * 100
                st.write(f"**{i+1}. {label}** ({prob_pct:.2f}%)")
                st.progress(float(top5_prob[i].item()))

# Educational resources section remains exactly the same
st.markdown("---")
st.header("📚 Presentation Talking Points")
with st.expander("🤔 What Actually Happened Here?"):
    st.write("The model did **not** 'see' the image the way a human does. It processed a grid of numbers through many mathematical layers, each one transforming those numbers slightly, until the final layer produced a probability for each of the 1,000 categories it was trained to recognize.\n\nThis also means the model can be **confidently wrong** — a high percentage score reflects how strongly the pattern matched something it learned during training, not a guarantee of correctness.")
with st.expander("🎯 Student Challenge"):
    st.write("Discuss as a class:\n1. What happens with a blurry image?\n2. What happens with multiple objects in the frame?\n3. Does the highest confidence score guarantee the prediction is correct?\n\n*This introduces an important idea: AI predictions are probabilistic, and probability is not the same as certainty.*")
with st.expander("💼 Interview Corner"):
    st.markdown("1. **What is transfer learning?** Why does it save time and data compared to training from scratch?\n2. **Why 224x224?** Why do we resize every image to a fixed size before feeding it to the network?\n3. **Preprocessing:** What does `preprocess_input` actually do to the pixel values?\n4. **Probabilities:** Why does the model output probabilities across 1,000 classes instead of one single, certain answer?\n5. **Limitations:** Name one real-world situation where a pretrained ImageNet model would perform poorly.")
