import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def main():
    # Load dataset
    train_df = pd.read_csv("sign_mnist_train.csv")
    test_df = pd.read_csv("sign_mnist_test.csv")

    y_train = train_df['label']
    X_train = train_df.drop('label', axis=1)
    y_test = test_df['label']
    X_test = test_df.drop('label', axis=1)

    X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.values.reshape(-1, 28, 28, 1) / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=25)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=25)

    # Build model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(25, activation='softmax')
    ])

    # Compile and train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model
    model.save("models/sign_language_model.h5")
    print("✅ Model saved as models/sign_language_model.h5")

if __name__ == "__main__":
    main()