import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

class FeatureExtractorFineTuner:
    def __init__(self, input_shape, num_classes, learning_rates):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rates = learning_rates
        self.feature_extractor = self.create_feature_extractor()
        self.fine_tuning_started = False
    
    def create_feature_extractor(self):
        # Load the DenseNet model pre-trained on ImageNet
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freeze the convolutional layer blocks and flatten layer
        for layer in base_model.layers:
            if 'conv' in layer.name or 'pool' in layer.name or 'flatten' in layer.name:
                layer.trainable = False
        
        # Get the output of the last convolutional layer
        output = base_model.output
        
        # Add a global average pooling layer
        output = tf.keras.layers.GlobalAveragePooling2D()(output)
        
        # Add a fully connected layer with softmax activation for classification
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(output)
        
        # Create the feature extraction model
        feature_extractor = Model(inputs=base_model.input, outputs=output)
        
        return feature_extractor
    
    def fine_tune_model(self):
        # Get the total number of layers in the model
        num_layers = len(self.feature_extractor.layers)
        
        # Create a list of optimizers with different learning rates for each layer
        optimizers = [SGD(learning_rate=lr) for lr in self.learning_rates]
        
        # Set the optimizers for each layer
        for i, optimizer in enumerate(optimizers):
            self.feature_extractor.layers[i].trainable = True
        
        # Compile the model with the final learning rate
        self.feature_extractor.compile(optimizer=optimizers[-1], loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, train_data, train_labels, epochs=10):
        if not self.fine_tuning_started:
            # Perform feature extraction
            optimizer = SGD(learning_rate=self.learning_rates[0])
            self.feature_extractor.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            self.feature_extractor.fit(train_data, train_labels, epochs=epochs)
            
            # Check the accuracy threshold to start fine-tuning
            accuracy = self.feature_extractor.history.history['accuracy'][-1]
            if accuracy >= 0.6 and accuracy <= 0.7:
                self.fine_tune_model()
                self.fine_tuning_started = True
        else:
            # Fine-tuning
            optimizer = SGD(learning_rate=self.learning_rates[-1])
            self.feature_extractor.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            self.feature_extractor.fit(train_data, train_labels, epochs=epochs)
