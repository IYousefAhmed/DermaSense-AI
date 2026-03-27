# 🧬 DermaSense AI

## AI-Powered Skin Lesion Analysis System

DermaSense AI is a full-stack AI-powered web application designed to analyze dermoscopic skin lesion images and provide automated classification with risk assessment.

The system integrates a deep learning model with an interactive web interface to deliver real-time predictions, confidence scores, and visual insights.

> ⚠️ Disclaimer: This system is intended for educational and research purposes only and does not replace professional medical diagnosis.

---

# 🚀 Live Demo

🔗 https://dermasense-ai-mxnxfmeng46zuwbdjqk9d7.streamlit.app/

---

# 🧠 System Overview

The system follows an end-to-end pipeline:

* User uploads image via web interface
* Image is validated and preprocessed
* AI model performs classification
* Results are visualized with probabilities
* Optional PDF report is generated

---

# 🔥 Key Features

* Real-time skin lesion classification
* Deep learning model (EfficientNet-B0)
* Confidence score & risk level
* Probability distribution chart
* Downloadable clinical-style PDF report
* Educational skin disease guide
* Multi-page interactive UI (Streamlit)

---

# 🧠 AI Model

* Architecture: EfficientNet-B0
* Framework: PyTorch
* Input Size: 224 × 224
* Output: 7 skin disease classes
* Activation: Softmax
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

# 🎯 Target Classes

* nv – Melanocytic Nevi
* mel – Melanoma
* bkl – Benign Keratosis
* bcc – Basal Cell Carcinoma
* akiec – Actinic Keratoses
* vasc – Vascular Lesion
* df – Dermatofibroma

---

# 📊 Dataset

HAM10000 Dataset

* 10,000 dermoscopic images
* 7 classification categories
* Data augmentation:

  * Rotation
  * Horizontal flipping
  * Normalization

---

# 🏗 System Architecture

The system consists of multiple components:

* Web Interface (Streamlit)
* Image Processing Module
* AI Model (EfficientNet)
* Visualization Module (Altair)
* Report Generator (PDF)

---

# 📂 Project Structure

```
DermaSense-AI/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Project_Info.py
│       ├── 2_Skin_Diseases_Guide.py
│       ├── 3_Model_Info.py
│       ├── 4_AI_Explanation.py
│       └── 5_How_to_Use.py
│
├── models/
│   ├── class_names.json
│   └── skin_cancer_efficientnet.pth
│
├── training/
│   └── train_model.py
│
├── static/
│   └── logo.png
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/IYousefAhmed/DermaSense-AI.git
cd DermaSense-AI
pip install -r requirements.txt
streamlit run app/Home.py
```

---

# 🔐 Security

* File type validation (JPG, PNG, JPEG)
* File size limitation
* Input validation
* No persistent image storage
* Error handling mechanisms

---

# 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

(Add your results here)

---

# 🌍 Deployment

* Streamlit Cloud
* Hugging Face Spaces
* GitHub (Version Control)

---

# 🔮 Future Enhancements

* REST API for integration
* User authentication system
* Prediction history tracking
* Explainable AI (Grad-CAM)
* Docker containerization
* CI/CD pipeline

---

# 👨‍💻 Author

**Yousef Ahmed**
AI & Cybersecurity Enthusiast

---

# ⭐ Contribution

Feel free to fork the project, open issues, or submit pull requests.

---

# 📌 Note

This project demonstrates the integration of Artificial Intelligence with web applications for medical image analysis and is intended as a practical implementation of applied deep learning systems.
