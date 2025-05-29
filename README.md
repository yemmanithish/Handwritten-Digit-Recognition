#🧠 Handwritten Digit Recognition

A deep learning project using a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0–9). Includes a **Tkinter-based GUI** to draw and predict digits live.

handwritten_digit_recognition/
├── data/
│ └── load_data.py # Loads and preprocesses the MNIST dataset
├── model/
│ └── cnn_model.py # CNN model definition
├── train/
│ └── train_model.py # Trains the model and saves weights
├── gui/
│ └── app.py # GUI for drawing and predicting digits
├── trained_model.h5 # Saved model weights after training
├── requirements.txt
└── README.md

📚 Dataset
  Uses the built-in MNIST dataset from TensorFlow:

  60,000 training images
  10,000 testing images
  Each image: 28x28 grayscale

🏗️ Model Architecture
  A simple CNN built using TensorFlow/Keras:

  Conv2D → ReLU → MaxPooling
  Conv2D → ReLU → MaxPooling
  Flatten → Dense → Dropout
  Output layer: 10 units (softmax)

🧪 Training
  Run this script to train and save the model:
  python -m train.train_model

🖌️ GUI: Draw and Predict Digits
Run the GUI application:
  python gui/app.py
🎯 Accuracy
  Expected test accuracy after training: ~98.5–99.0%

📝 Example Output
  Epoch 5/5
  1688/1688 [==============================] - 12s - loss: 0.0153 - accuracy: 0.9950
  313/313 [==============================] - 1s - loss: 0.0411 - accuracy: 0.9875
  Test Accuracy: 0.9875

🚀 Future Improvements
  Export predictions to CSV
  Deploy on web using Flask or Streamlit
  Real-time digit prediction while drawing
  Support for uploaded images

🧑‍💻 Author
  Nithish Yemman
  GitHub: @yemmanithish
