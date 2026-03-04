# 🧬 DermaSense AI

## Clinical Skin Lesion Risk Analysis using Deep Learning (ResNet50)

DermaSense AI is a production-ready, AI-powered web application designed to classify dermoscopic skin lesion images using a fine-tuned ResNet50 deep learning model.

The system provides automated risk assessment with confidence scoring and probability distribution visualization to support early-stage diagnostic assistance.

> ⚠️ This system is intended for research and educational purposes only. It does not replace professional medical diagnosis.

---

# 🚀 Live Demo

(Coming Soon – Public Deployment Link)

---

# 🧠 Model Overview

The model is based on **ResNet50 (Transfer Learning)** with a custom classification head.

### Architecture Details:
- Backbone: ResNet50 (pretrained on ImageNet)
- Input Size: 224 × 224
- Output Classes: 7 (HAM10000 dataset)
- Activation: Softmax
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Training Strategy: Feature Extraction (Frozen Backbone)

### Target Classes:
- nv – Melanocytic Nevi
- mel – Melanoma
- bkl – Benign Keratosis
- bcc – Basal Cell Carcinoma
- akiec – Actinic Keratoses
- vasc – Vascular Lesion
- df – Dermatofibroma

---

# 📊 Dataset

Dataset used: **HAM10000 – Human Against Machine with 10000 training images**

- 7 skin lesion categories
- Stratified train/validation/test split
- Data augmentation applied:
  - Horizontal flipping
  - Rotation
  - Normalization

---

# 🔐 Security Considerations

- File type validation (JPG / PNG)
- Maximum file size limit (5MB)
- Image integrity verification
- No permanent image storage
- Secure model loading

---

# 🛠 Tech Stack

- Python 3.10+
- PyTorch
- Torchvision
- Streamlit
- Pandas
- NumPy
- Altair
- Git & GitHub

---

# 📂 Project Structure
DermaSense-AI/
│
├── app/
│ └── app.py
│
├── models/
│ └── skin_cancer_resnet50.pth
│
├── training/
│ └── train_model.py
│
├── static/
│ └── logo.png
│
├── docs/
│
├── requirements.txt
└── README.md


---

# ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/DermaSense-AI.git
cd DermaSense-AI
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
streamlit run app/app.py

The application will run locally at:

http://localhost:8501
📈 Model Performance

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

(You can add your final accuracy value here.)

🌍 Deployment

This application can be deployed using:

Streamlit Cloud

Render

VPS + Nginx

Docker containerization

🔮 Future Enhancements

User authentication system

Prediction history tracking

PDF medical-style report generation

REST API endpoint

Docker support

CI/CD integration

Cloud deployment with custom domain

👨‍💻 Author

Developed by Yousef
AI & Cybersecurity Enthusiast