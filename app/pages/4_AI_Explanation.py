import streamlit as st

st.title("🧠 AI Explanation")

st.subheader("How the AI System Works")

st.write("""
DermaSense AI uses a deep learning model to analyze skin lesion images
and classify them into one of seven skin disease categories.
""")

st.subheader("Step 1: Image Upload")

st.write("""
The user uploads a dermoscopic image of a skin lesion.
The system checks the file and prepares it for analysis.
""")

st.subheader("Step 2: Image Preprocessing")

st.write("""
The image is resized and normalized using image transformation
techniques to match the model input size (224x224 pixels).
""")

st.subheader("Step 3: Deep Learning Model")

st.write("""
The system uses an EfficientNet-B0 convolutional neural network
trained on the HAM10000 skin lesion dataset.
""")

st.subheader("Step 4: Prediction")

st.write("""
The AI model analyzes the image and outputs the predicted
skin disease class along with a confidence score.
""")