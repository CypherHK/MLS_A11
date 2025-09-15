# in conda mls-trans, because of the version of tf and keras, set as 2.16 and 3.11
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def create_model():
    """
    Create the CNN model for MNIST classification.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_and_preprocess_data():
    """
    Load and preprocess MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)  # shape: (60000, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)    # shape: (10000, 28, 28, 1)

    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the model and evaluate performance.
    """
    history = model.fit(
        x_train, y_train,
        epochs=5,
        validation_data=(x_test, y_test)
    )
    return history


def plot_history(history):
    """
    Plot training & validation accuracy and loss curves.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
   
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    model = create_model()
    history = train_model(model, x_train, y_train, x_test, y_test)

    # 可视化训练过程
    plot_history(history)

    # Save
    model.save("mnist_cnn_model.keras")
    model.export("mnist_cnn_savedmodel")
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
