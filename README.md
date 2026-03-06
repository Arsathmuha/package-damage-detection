📦 Package Quality Detector

An AI-powered system that detects whether a package is Good or Damaged using deep learning and computer vision.

The project uses Transfer Learning with ResNet50 and provides an interactive Streamlit web application where users can upload a package image and get a prediction.

🚀 Features

Detects Good vs Damaged packages

Uses deep learning with transfer learning

Image data augmentation for better generalization

Interactive Streamlit web interface

Displays confidence score for predictions

Training accuracy and loss visualization

🧠 Model Architecture

The model uses:

ResNet50

Pretrained on ImageNet

Fine-tuned for binary classification

Pipeline
Package Image
      ↓
Image Preprocessing
      ↓
ResNet50 Feature Extraction
      ↓
Fully Connected Layers
      ↓
Prediction: Good / Damaged
📂 Project Structure
Package-Quality-Detector
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── predict.py              # Command line prediction script
├── check.py                # Utility script
├── requirements.txt        # Python dependencies
├── package_quality_model.h5 # Trained model
├── training_history.png    # Training accuracy & loss plot
│
└── dataset/
     ├── train/
     │    ├── good/
     │    └── damaged/
     │
     └── validation/
          ├── good/
          └── damaged/
📊 Training Results

The training process produces accuracy and loss graphs.

Example:

training_history.png

This helps evaluate model performance and detect overfitting.

📂 Dataset

The dataset used for training is available here:

🔗 Dataset Link:
https://www.kaggle.com/datasets/erteteyhvbe/dataset

After downloading the dataset, organize it in the following structure:

dataset/
   train/
      good/
      damaged/
   validation/
      good/
      damaged/
⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/package-quality-detector.git
cd package-quality-detector

Install dependencies:

pip install -r requirements.txt
▶️ Run the Web App

Start the Streamlit application:

streamlit run app.py

Open in browser:

http://localhost:8501

Upload a package image to get the prediction.

🧪 Command Line Prediction

You can also test images directly:

python predict.py path/to/image.jpg

Example output:

Prediction : Good
Confidence : 92.31 %
📈 Future Improvements

Possible upgrades:

Detect damage location using object detection

Improve accuracy with EfficientNet

Deploy the model as an online API

Add real-time camera inspection

Expand dataset with more damage types

🛠 Tech Stack

Python

TensorFlow / Keras

Streamlit

NumPy

Pillow

👨‍💻 Author

Developed by Mohamed Arsath

⭐ Support

If you find this project useful, please ⭐ the repository.
