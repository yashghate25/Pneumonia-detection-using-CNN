# Pneumonia Detection using CNN 🫁🧠

This project uses a Convolutional Neural Network (CNN) model to detect pneumonia from chest X-ray images. It is designed to assist radiologists and medical professionals in faster and more accurate diagnosis of pneumonia using deep learning techniques.
---
## 📁 Project Structure
MINI-PROJECT/
│
├── model/
│ └── pneumonia.h5 # Trained CNN model
│
├── dataset/
│ ├── train/
│ ├── test/
│ └── val/ # Chest X-ray images from Kaggle
│
├── notebooks/
│ └── Pneumonia_Classifier.ipynb # Jupyter notebook for training & evaluation
│
├── app/
│ ├── app.py # Streamlit or Flask app for prediction
│ ├── templates/
│ └── static/
│
├── requirements.txt # Required Python packages
├── README.md # This file
└── LICENSE


---

## 🧠 Model Used

- **Model:** Convolutional Neural Network (CNN)
- **Architecture:** 4 Conv layers + MaxPooling + Flatten + Dense layers
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, AUC
- **Final Model:** `pneumonia.h5`

You can also replace CNN with **DenseNet-169** or ResNet for better accuracy.

---

## 📂 Dataset

- Dataset Source: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Images are divided into:
  - `train/`
  - `test/`
  - `val/`

> The dataset contains chest X-ray images categorized as **Normal** and **Pneumonia**.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yashghate25/Pneumonia-detection-using-CNN.git
cd Pneumonia-detection-using-CNN

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate      # For Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app (for Streamlit or Flask)
python app/app.py
