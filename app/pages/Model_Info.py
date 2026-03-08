import streamlit as st

st.title("AI Model Information")

st.subheader("Model Architecture")

st.write("""
The model used in this system is **EfficientNet-B0**,
a convolutional neural network architecture designed
for efficient image classification.
""")

st.subheader("Framework")

st.write("""
The system was built using:

• PyTorch  
• TorchVision  
• Streamlit  
""")

st.subheader("Dataset")

st.write("""
The model was trained on the **HAM10000 dataset**,
a well-known dermatology dataset containing thousands
of dermoscopic skin lesion images.
""")

st.subheader("Classes")

st.write("""
The model predicts 7 skin lesion categories:

• akiec  
• bcc  
• bkl  
• df  
• mel  
• nv  
• vasc
""")