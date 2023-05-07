"""
Call the model from model.py folder using #from model import FeatureExtractorFineTuner


By default, Epoch=10, and I trained it for 20 epochs to test. But, you probably want to run it for 40-(close to 100) epochs at least to receive a high accuracy
score.

"""




import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class labels to one-hot encoded vectors
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the input shape for the feature extractor
input_shape = x_train.shape[1:]

# Define the learning rates for fine-tuning
learning_rates = [0.001, 0.001, 0.0001]

# Create an instance of FeatureExtractorFineTuner
extractor_fine_tuner = FeatureExtractorFineTuner(input_shape, num_classes, learning_rates)

# Train the model
extractor_fine_tuner.train(x_train, y_train, epochs=20)


