# Hand Gesture Recognition using ASL and CNN
This project is a deep learning-based **Hand Gesture Recognition System** that uses **Convolutional Neural Networks (CNNs)** to classify American Sign Language (ASL) alphabets from real hand images.

## 📌 Project Overview
The goal of this project is to recognize ASL alphabets using a trained CNN model. It takes hand gesture images as input and predicts the corresponding alphabet.

## 🚀 Features
- Trained on the **ASL Alphabet Dataset** (real hand images).
- Uses **TensorFlow/Keras CNN** architecture.
- Real-time prediction from webcam (optional).
- Clean and modular code with separate scripts for training and prediction.

## 🧠 Model Architecture
- 3 Convolutional Layers with ReLU and MaxPooling  
- Flatten and Dense Layers  
- Dropout for regularization  
- Trained with `Adam` optimizer and `categorical_crossentropy` loss

## 📁 Directory Structure
hand-gesture-recognition/

├── data/ # ASL dataset files (images)

├── models/ # Saved trained CNN models

├── train.py # Script to train the CNN model

├── predict.py # Script for real-time or batch prediction

├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 💾 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
2.Install required packages:

    pip install -r requirements.txt
   
## ▶️ Usage
To train the model, run:
  
  python train.py
To run real-time gesture recognition using your webcam:
  
  python predict.py
  
## 🎯 Results
- The model currently achieves moderate accuracy on the ASL Alphabet dataset, with some misclassifications.
- Real-time prediction works but may require further tuning and more training data for improved accuracy.
- This project serves as a solid foundation for future improvements like better preprocessing, data augmentation, and more complex models.


## 📚 References
- **ASL Alphabet Dataset:**  
  [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)

- **TensorFlow CNN Tutorial:**  
  [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)

- **OpenCV Documentation:**  
  [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)


