from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import img_to_array


import numpy as np
import os
import cv2
import json

from SETTINGS import *


def createModel(inputShape):
    # Input layer
    input_img = Input(inputShape)

    # Convolutional layers


    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # Shared dense layers
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Single output layer for all holes
    output = Dense(4 * 3, activation='linear')(x)  # 4 holes * (x, y, existence)
    output = Reshape((4, 3))(output)  # Reshape to 4x3 for clarity

    model = Model(inputs=input_img, outputs=output)

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.summary()

    return model


def flip_image(image, labels):
    """
    Flip the image and labels horizontally.
    """
    flipped_image = cv2.flip(image, 1)  # 1 means horizontal flip
    
    flipped_labels = []
    for label in labels:
        x, y, exist = label
        flipped_label = [-x, y, exist]  # Flip x coordinate
        flipped_labels.append(flipped_label)
        
    return flipped_image, flipped_labels

def rotate_image_90_degrees(image, labels):
    """
    Rotate the image by 90 degrees counter-clockwise and adjust labels accordingly.
    """
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    rotated_labels = []
    for label in labels:
        x, y, exist = label
        # For a 90 degree rotation: new_x = -y, new_y = x
        rotated_label = [y, -x, exist]
        rotated_labels.append(rotated_label)
        
    return rotated_image, rotated_labels





# Load labels from JSON
def load_labels(dataPath, image_width, image_height):

    episodes = os.listdir(dataPath)

    labels = []

    for episode in episodes:
        epsisode_path = os.path.join(dataPath, episode)
        json_path = os.path.join(epsisode_path, "labels.json")
        


        with open(json_path, 'r') as file:
            data = json.load(file)
        
        for item in data['images']:
            label = []
            for i in range(1, 5):

                hole = item[str(i)]


                x = (hole['x'] / image_width) * 2 - 1  # Normalize x to [-1, 1]
                y = (hole['y'] / image_height) * 2 - 1  # Normalize y to [-1, 1]
                exist = hole['existance']
                if exist is 0:
                    x = 0
                    y = 0
                    
                label.append([x, y, exist])

            
            #print(label)
            labels.append(label)
    return np.array(labels)

# Load and preprocess images
def load_images(dataPath, imageSize):


    episodes = os.listdir(dataPath)
    images = []
    realImageSize=None

    for episode in episodes:
        image_folder = os.path.join(dataPath, episode)

        imageNames = os.listdir(image_folder)
        #remove all files that are not images
        imageNames = [image for image in imageNames if image.endswith(".png") or image.endswith(".jpg")]


        imageNames.sort(key=lambda x: int(x.split("_")[0].split("e")[-1]))


        
        for name in imageNames:

            path = os.path.join(image_folder, name)
            
            image = cv2.imread(path)

            if realImageSize is None:
                realImageSize = image.shape[:2]

            image = cv2.resize(image, imageSize)

            image = img_to_array(image) / 255.0  # Normalize to [0, 1]
            images.append(image)
    return np.array(images), realImageSize




def augment_data(images, labels):
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(images, labels):
        # Original image and labels
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Flipped image and labels
        flipped_image, flipped_labels = flip_image(image, label)
        augmented_images.append(flipped_image)
        augmented_labels.append(flipped_labels)
        
        # Rotated image and labels

        #all four rotations for both flipped and non flipped image
        for i in range(3):
            image, label = rotate_image_90_degrees(image, label)
            augmented_images.append(image)
            augmented_labels.append(label)
            
            flipped_image, flipped_labels = rotate_image_90_degrees(flipped_image, flipped_labels)
            augmented_images.append(flipped_image)
            augmented_labels.append(flipped_labels)
        
    return np.array(augmented_images), np.array(augmented_labels)

def train_model(model, images, labels, epochs=20):

    #augment the data

    model.fit(images, labels, epochs=epochs)

    return model








def main():
    
    path="Training/Data/PathData"
    
    #extract episodes from path
    episodes = os.listdir(path)

    
    




    
    images, realImageSize = load_images(path, IMAGE_SIZE)

    labels = load_labels(path, realImageSize[0], realImageSize[1])


    images, labels = augment_data(images, labels)

    #shuffle the images and labels
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    #split the data into training and validation data
    split = int(0.8 * len(images))
    train_images = images[:split]
    train_labels = labels[:split]
    val_images = images[split:]
    val_labels = labels[split:]



    input_shape = images[0].shape

    model = createModel(input_shape)

    model = train_model(model, train_images, train_labels)

    #save the model
    model.save("Training/model.keras")



if __name__ == "__main__":
    main()



