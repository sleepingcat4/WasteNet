from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

def augment_dataset_from_folder(data_folder, target_size, batch_size):
    # Create an instance of the ImageDataGenerator
    datagen = ImageDataGenerator(
        zoom_range=0.2,                 # Randomly zoom images
        rotation_range=20,              # Randomly rotate images
        width_shift_range=0.2,          # Randomly shift images horizontally
        height_shift_range=0.2,         # Randomly shift images vertically
        horizontal_flip=True,           # Randomly flip images horizontally
        fill_mode='nearest'             # Fill new pixels with nearest surround pixels
    )
    
    augmented_images = []
    augmented_labels = []
    
    # Iterate over the subfolders in the data folder
    for class_folder in os.listdir(data_folder):
        class_path = os.path.join(data_folder, class_folder)
        
        # Iterate over the images in each class folder
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            # Load the image
            image = load_img(image_path, target_size=target_size)
            image_array = img_to_array(image)
            
            # Reshape the image to (1, height, width, channels) for augmentation
            image_array = image_array.reshape((1,) + image_array.shape)
            
            # Generate augmented images
            augmented = datagen.flow(image_array, batch_size=batch_size)
            
            # Retrieve augmented images and labels
            for i in range(batch_size):
                augmented_image = augmented.next()[0]
                
                # Append augmented image and label to the augmented dataset
                augmented_images.append(augmented_image)
                augmented_labels.append(class_folder)
    
    # Convert the augmented images and labels to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    return augmented_images, augmented_labels
