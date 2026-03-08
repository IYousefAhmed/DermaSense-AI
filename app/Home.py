import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
import os
import base64
import io
import json
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="DermaSense AI",
    page_icon="🧬",
    layout="wide"
)
st.title("Home - AI Skin Analysis")

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,"..","models","skin_cancer_efficientnet.pth")

CLASS_PATH = os.path.join(BASE_DIR,"..","models","class_names.json")

LOGO_PATH = os.path.join(BASE_DIR,"..","static","logo.png")

# =====================================
# LOAD CLASS NAMES
# =====================================

with open(CLASS_PATH,"r") as f:
    class_names = json.load(f)

# =====================================
# DARK UI STYLE
# =====================================

st.markdown("""
<style>

body{
background-color:#020617;
color:white;
}

.title{
text-align:center;
font-size:52px;
font-weight:800;
color:white;
margin-top:10px;
}

.subtitle{
text-align:center;
font-size:20px;
color:#94a3b8;
margin-bottom:30px;
}

.logo{
display:flex;
justify-content:center;
margin-bottom:10px;
}

.footer{
text-align:center;
margin-top:40px;
color:#64748b;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

if os.path.exists(LOGO_PATH):

    with open(LOGO_PATH,"rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
    f"""
    <div class="logo">
        <img src="data:image/png;base64,{encoded}" width="230">
    </div>
    """,
    unsafe_allow_html=True
    )

st.markdown('<div class="title">DermaSense AI</div>',unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">AI Powered Skin Lesion Analysis System</div>',
unsafe_allow_html=True
)

st.markdown("---")

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = torch.nn.Linear(num_features,len(class_names))

    model.load_state_dict(torch.load(MODEL_PATH,map_location=device))

    model.to(device)

    model.eval()

    return model,device

model,device = load_model()

# =====================================
# IMAGE TRANSFORM
# =====================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =====================================
# CLASS DETAILS
# =====================================

class_details = {

"mel":{"name":"Melanoma","risk":"High"},
"bcc":{"name":"Basal Cell Carcinoma","risk":"Medium"},
"akiec":{"name":"Actinic Keratosis","risk":"Medium"},
"nv":{"name":"Melanocytic Nevus","risk":"Low"},
"bkl":{"name":"Benign Keratosis","risk":"Low"},
"vasc":{"name":"Vascular Lesion","risk":"Low"},
"df":{"name":"Dermatofibroma","risk":"Low"}

}

# =====================================
# PDF REPORT
# =====================================

def generate_pdf(label,confidence):

    buffer = io.BytesIO()

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(buffer)

    elements = []

    elements.append(
    Paragraph("DermaSense AI Clinical Report",styles["Heading1"])
    )

    elements.append(Spacer(1,20))

    elements.append(
    Paragraph(
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    styles["Normal"]
    )
    )

    elements.append(Spacer(1,20))

    elements.append(
    Paragraph(
    f"Prediction: {class_details[label]['name']}",
    styles["Normal"]
    )
    )

    elements.append(
    Paragraph(
    f"Risk Level: {class_details[label]['risk']}",
    styles["Normal"]
    )
    )

    elements.append(
    Paragraph(
    f"Confidence Score: {confidence:.2f}%",
    styles["Normal"]
    )
    )

    elements.append(Spacer(1,40))

    elements.append(
    Paragraph(
    "Disclaimer: This AI system is for educational purposes only.",
    styles["Normal"]
    )
    )

    doc.build(elements)

    buffer.seek(0)

    return buffer

# =====================================
# MAIN LAYOUT
# =====================================

left,right = st.columns([1,1])

# =====================================
# IMAGE UPLOAD (SECURE VERSION)
# =====================================

with left:

    st.subheader("Upload Skin Image")

    uploaded_file = st.file_uploader(
    "Upload dermoscopic image",
    type=["jpg","png","jpeg"]
    )

    image = None

    if uploaded_file:

        # منع الملفات الكبيرة
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File too large. Maximum size is 5MB.")

        else:

            try:

                # التأكد ان الملف صورة حقيقية
                image = Image.open(uploaded_file)
                image.verify()

                image = Image.open(uploaded_file).convert("RGB")

                st.image(image,width=420)

            except:
                st.error("Invalid image file. Please upload a valid image.")

# =====================================
# AI ANALYSIS
# =====================================

with right:

    if image is not None:

        st.subheader("AI Analysis")

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(input_tensor)

            probs = torch.softmax(output,dim=1)

        prob_values = probs[0].cpu().numpy()

        idx = np.argmax(prob_values)

        label = class_names[idx]

        confidence = prob_values[idx]*100

        risk = class_details[label]["risk"]

        if risk == "High":
            color = "#ef4444"
        elif risk == "Medium":
            color = "#f59e0b"
        else:
            color = "#22c55e"

        st.markdown(f"""
        <div style="
        padding:25px;
        border-radius:15px;
        background:#111827;
        border:2px solid {color};
        box-shadow:0 0 20px {color}40;
        ">

        <h2 style="margin-bottom:10px;color:white;">
        🧬 Prediction: {class_details[label]["name"]}
        </h2>

        <p style="font-size:18px;color:#cbd5e1;">
        Risk Level:
        <span style="color:{color};font-weight:bold;font-size:20px;">
        {risk}
        </span>
        </p>

        <p style="font-size:18px;color:#cbd5e1;">
        Confidence Score:
        <b style="color:white;">{confidence:.2f}%</b>
        </p>

        </div>
        """,unsafe_allow_html=True)

        st.progress(float(confidence)/100)

        df = pd.DataFrame({
            "Class":class_names,
            "Probability":prob_values*100
        })

        df["Risk"] = df["Class"].map(lambda x: class_details[x]["risk"])

        chart = alt.Chart(df).mark_bar(
            cornerRadiusEnd=6,
            size=35
        ).encode(

            x=alt.X(
                "Probability:Q",
                title="Probability (%)",
                axis=alt.Axis(labelColor="white",titleColor="white")
            ),

            y=alt.Y(
                "Class:N",
                sort="-x",
                axis=alt.Axis(labelColor="white",titleColor="white")
            ),

            color=alt.Color(
                "Risk:N",
                scale=alt.Scale(
                    domain=["High","Medium","Low"],
                    range=["#ef4444","#f59e0b","#22c55e"]
                ),
                legend=alt.Legend(
                    title="Risk Level",
                    labelColor="white",
                    titleColor="white"
                )
            ),

            tooltip=["Class","Probability","Risk"]

        ).properties(
            height=320
        ).configure_view(
            stroke=None
        )

        st.altair_chart(chart,use_container_width=True)

        pdf = generate_pdf(label,confidence)

        st.download_button(
        "Download Medical Report",
        pdf,
        "DermaSense_Report.pdf",
        "application/pdf"
        )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
"""
<div class="footer">
DermaSense AI • AI Dermatology Assistant
</div>
""",
unsafe_allow_html=True
)