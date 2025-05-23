import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
    # Load the pre-trained model
    model = load_model("models/sign_language_model.h5")

    # Label map (25 classes, A-Y excluding J and Z)
    labels = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
        'T', 'U', 'V', 'W', 'X', 'Y'
    ]

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Region of Interest
        roi = frame[100:400, 100:400]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(reshaped)
        index = np.argmax(pred)
        predicted_label = labels[index] if index < len(labels) else "Unknown"

        # Show prediction
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
        cv2.putText(frame, f'Prediction: {predicted_label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
