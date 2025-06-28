# ♻️ Waste Classifier App

An AI-powered image classification app that detects and categorizes waste materials (e.g., plastic, paper, metal, glass) using a deep learning model trained on a real-world garbage dataset. Built with TensorFlow and Streamlit, this project aims to promote smart waste segregation and environmental awareness.

---

## 📌 Project Objective

Manual waste segregation is often inefficient and inconsistent. This project aims to:

- 🗑️ Automatically classify images of waste into predefined categories
- 🧠 Use a deep learning model (MobileNetV2) to ensure high accuracy
- 🌐 Provide an easy-to-use web interface for real-time classification
- 🌍 Encourage sustainable waste management and recycling behavior

---

## 🚀 Features

✅ Upload an image of any waste item  
✅ Real-time prediction of waste type with confidence score  
✅ MobileNetV2-based model: lightweight & fast  
✅ Simple Streamlit UI for interaction  
✅ Reproducible ML pipeline (train + predict)  
✅ Dataset & code available for educational reuse  

---

## 📂 Dataset

This project uses the **[Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)** from Kaggle.

> 📦 **Note**: Due to GitHub storage limits, the dataset is not included directly in this repository.

### 🔗 Download the dataset here:
[https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

### 💾 Setup:

1. Download and extract the dataset
2. Rename the folder to: `garbage-dataset`
3. Place it inside the project directory like this:
waste_classifier/garbage-dataset/

shell
Copy
Edit

### 📁 Folder Structure:
garbage-dataset/
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
└── trash/

yaml
Copy
Edit

---

## 🧠 Model Architecture

The model is built using **MobileNetV2** as the feature extractor:

- 📐 Input: 224x224 RGB image
- 🔍 Base: Pretrained MobileNetV2 (ImageNet weights)
- 🔄 Global Average Pooling
- 🔚 Dense layer with softmax (10 output classes)

> Training was done using `ImageDataGenerator` with data augmentation and 80/20 train-validation split.

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Muhsina-Rinsha-TP/waste_classifier.git
   cd waste_classifier
Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run app.py

![Demo Screenshot](demo.png)




📁 Folder Structure
bash
Copy
Edit
waste_classifier/
├── app.py                 # Streamlit app for prediction
├── train_model.py         # Model training script
├── waste_model.h5         # Saved Keras model
├── labels.txt             # Class labels
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignore unnecessary files
├── README.md              # This file!
├── garbage-dataset/       # Training dataset (external)
└── demo.png               # App screenshot 
🧠 Future Improvements
🔍 Add image preprocessing for low-quality/mixed waste detection

💬 Add a voice-based assistant for accessibility

📊 Show model confidence heatmaps (Grad-CAM)

☁️ Deploy using Docker, Streamlit Cloud, or Hugging Face Spaces

📜 License
This project is licensed under the MIT License — feel free to fork and modify.

🙌 Acknowledgements
Dataset by: Kaggle - Garbage Classification

TensorFlow/Keras for deep learning

Streamlit for easy web deployment

Made with ❤️ for AI4Environment | Muhsina Rinsha