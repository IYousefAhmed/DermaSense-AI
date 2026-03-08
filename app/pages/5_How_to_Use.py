import streamlit as st

st.title("📋 How to Use")

st.subheader("Step 1: Upload Image")

st.write("""
Upload a dermoscopic skin image using the upload button
on the main page.
""")

st.subheader("Step 2: AI Analysis")

st.write("""
The AI model automatically analyzes the uploaded image
and predicts the most likely skin lesion type.
""")

st.subheader("Step 3: View Results")

st.write("""
The system displays:

• Predicted skin disease  
• Risk level  
• Confidence score  
• Probability chart
""")

st.subheader("Step 4: Download Report")

st.write("""
Users can download a medical-style report containing
the prediction and analysis results.
""")