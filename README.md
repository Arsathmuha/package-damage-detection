# 📦 Package Quality Detector

An AI-powered system that detects whether a **package is Good or Damaged** using deep learning and computer vision.

The project uses **Transfer Learning with ResNet50** and provides an interactive **Streamlit web application** where users can upload a package image and get a prediction.

---

# 🚀 Features

* Detects **Good vs Damaged packages**
* Uses **deep learning with transfer learning**
* Image **data augmentation for better generalization**
* Interactive **Streamlit web interface**
* Displays **confidence score for predictions**
* Training accuracy and loss visualization

---

# 🧠 Model Architecture

The model uses **ResNet50** pretrained on **ImageNet** and fine-tunes it for binary classification.

Pipeline:

Package Image
↓
Image Preprocessing
↓
ResNet50 Feature Extraction
↓
Fully Connected Layers
↓
Prediction (Good / Damaged)

---

# 📂 Project Structure

```
Package-Quality-Detector
│
├── app.py
├── train_model.py
├── predict.py
├── check.py
├── requirements.txt
├── package_quality_model.h5
├── training_history.png
│
└── dataset/
     ├── train/
     │    ├── good/
     │    └── damaged/
     │
     └── validation/
          ├── good/
          └── damaged/
```

---

# 📊 Training Results

The model training produces accuracy and loss graphs saved as:

```
training_history.png
```

These graphs help evaluate model performance and detect overfitting.

---

# 📂 Dataset

The dataset used for training can be downloaded from the link below:

DATASET LINK:
https://www.kaggle.com/datasets/erteteyhvbe/dataset

After downloading, organize the dataset like this:

```
dataset/
   train/
      good/
      damaged/
   validation/
      good/
      damaged/
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Arsathmuha/package-quality-detector.git
cd package-quality-detector
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Web App

Start the Streamlit application:

```
streamlit run app.py
```

Open in your browser:

```
http://localhost:8501
```

Upload a package image to get the prediction.

---

# 🧪 Command Line Prediction

You can also run predictions using the terminal:

```
python predict.py path/to/image.jpg
```

Example output:

```
Prediction : Good
Confidence : 92.31%
```

---

# 🛠 Tech Stack

Python
TensorFlow / Keras
Streamlit
NumPy
Pillow

---

# 👨‍💻 Author

Mohamed Arsath

---

# ⭐ Support

If you find this project useful, please consider giving it a ⭐ on GitHub.
