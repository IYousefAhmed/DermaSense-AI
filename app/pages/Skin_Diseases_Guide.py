import streamlit as st

st.title("Skin Diseases Guide")

st.write("Overview of the seven skin lesion classes used in the AI model.")

diseases = {
"Melanoma (mel)": "Most dangerous skin cancer. Requires urgent medical attention.",
"Basal Cell Carcinoma (bcc)": "Common skin cancer but rarely spreads.",
"Actinic Keratosis (akiec)": "Pre-cancerous lesion caused by sun damage.",
"Benign Keratosis (bkl)": "Non-cancerous skin growth often seen in adults.",
"Dermatofibroma (df)": "Benign skin nodule usually harmless.",
"Vascular Lesion (vasc)": "Lesions related to blood vessels.",
"Melanocytic Nevus (nv)": "Common mole. Usually harmless."
}

for name,desc in diseases.items():
    st.subheader(name)
    st.write(desc)