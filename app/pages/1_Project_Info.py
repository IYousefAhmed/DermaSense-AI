import streamlit as st

st.title("📘 Project Information")

st.subheader("About The Project")

st.write("""
DermaSense AI is a deep learning system designed to assist in
the analysis of skin lesions using artificial intelligence.

The system uses a trained EfficientNet-B0 model to classify
dermoscopic skin images into seven different diagnostic categories.
""")

st.subheader("Project Goals")

st.write("""
• Assist early detection of skin cancer  
• Demonstrate AI in medical imaging  
• Provide a simple interface for skin lesion analysis
""")

st.subheader("Features")

st.write("""
• Upload dermoscopic skin images  
• AI powered classification  
• Risk level assessment  
• Confidence score visualization  
• PDF medical report generation
""")