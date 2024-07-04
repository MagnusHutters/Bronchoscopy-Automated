

import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape
import numpy as np
import json


from Training.EpisodeLoader import *

from Training.SETTINGS import *


def imageLoader(image_path):
    tf.print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0  # Convert img to float32 and normalize
    return img

def create_model():

    image_input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    pathInputShape = (4, 4)
    statesInputShape = (1,2)




    # Define the inputs
    image_input = Input(shape=image_input_shape, name='image_input')
    pathInput = Input(shape=pathInputShape, name='pathInput')
    statesInput = Input(shape=statesInputShape, name='statesInput')

    # Define the CNN model for image input
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flattenCNN = Flatten()(pool2)


    # Define Dense layers for other inputs
    pathDense = Dense(16, activation='relu')(pathInput)
    statesDense = Dense(4, activation='relu')(statesInput)

    #statesDense = Reshape((4,1))(statesDense)


    #predictionsDense = Dense(16, activation='relu')(predictionsInput)
    #flatten the dense layers
    pathDense = Flatten()(pathDense)
    statesDense = Flatten()(statesDense)
    #predictionsDense = Flatten()(predictionsDense)


    concatenatedArrays = concatenate([pathDense, statesDense])
    #another dense
    other = Dense(64, activation='relu')(concatenatedArrays)

    #other = Flatten(input_shape=(4,64))(other)
    #other = Reshape((256,))(other)


    # Concatenate all inputs
    concatenated_inputs = concatenate([flattenCNN, other])

    # Dense layer to combine features
    combined_features = Dense(128, activation='relu')(concatenated_inputs)

    # Dense layer to combine features
    combined_features = Dense(64, activation='relu')(combined_features)

    # Output layer - output is a 3-element vector representing output in range -1 to 1
    output = Dense(3, activation='linear')(combined_features)

    # Define the model
    
    
    model = tf.keras.Model(inputs=[image_input, pathInput, statesInput], outputs=output)
    #model = tf.keras.Model(inputs=image_input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Print model summary
    model.summary()


    return model








def train():


    #load data
    path = "Training/Data/BronchoData"

    episodes = os.listdir(path)

    images = []
    inputs = []
    states = []
    paths = []
    #predictions = []

    for episode in episodes:
        episodePath = os.path.join(path, episode)
        episodePath = correctPathForOS(episodePath)
        

        episodeImages, episodeInputs, episodeStates, episodePaths, episodePredictions = loadEpisodeFull(episodePath)
        images.extend(episodeImages)
        inputs.extend(episodeInputs)
        states.extend(episodeStates)
        paths.extend(episodePaths)
        #predictions.extend(episodePredictions)


    #check if the data is loaded correctly and the shapes are correct
    #to numpy array
    images = np.array(images)
    inputs = np.array(inputs)
    states = np.array(states)
    paths = np.array(paths)
    #shapes
    print(f"Images: {images.shape}")
    print(f"Inputs: {inputs.shape}")
    print(f"States: {states.shape}")
    print(f"Paths: {paths.shape}")

    #wait for user input to continue
    #input("Press Enter to continue...")
    #for i in range(len(images)):
        #load image and display the shape
        #image = cv2.imread(images[i])
        #shape = image.shape

        #print(f"No: {i:4d} ImageName: {images[i]} Input: {inputs[i].shape} Image: {images[i].shape} imageShape: {shape} State: {states[i].shape} Path: {paths[i].shape}")
    

    
    #input("Press Enter to continue...")

    print(f"Creating dataset...")

    imageDataset = tf.data.Dataset.from_tensor_slices(images)
    stateDataset = tf.data.Dataset.from_tensor_slices(states)
    pathDataset = tf.data.Dataset.from_tensor_slices(paths)
    labelInput = tf.data.Dataset.from_tensor_slices(inputs)

    print(f"Zipping dataset...")
    combined_dataset = tf.data.Dataset.zip((imageDataset, pathDataset, stateDataset, labelInput))
    #combined_dataset = tf.data.Dataset.zip((imageDataset, labelInput))




    print(f"Mapping dataset...")
    combined_dataset = combined_dataset.map(
        lambda img, arr1, arr2, label: (
            (img, arr1, arr2), label
            )
    )

    




    # combined_dataset = combined_dataset.map(
    #     lambda img_path, label: (
    #         img_path, label
    #         )
    # )
    

    buffer_size = 1000  # This is just an example; adjust based on your dataset size and memory constraints
    batch_size = 32
    print(f"Shuffling and batching dataset...")
    combined_dataset = combined_dataset.shuffle(buffer_size)


    dataSetSize = len(images)
    trainSize = int(dataSetSize * 0.9)
    valSize = dataSetSize - trainSize


    train_dataset = combined_dataset.take(trainSize)
    val_dataset = combined_dataset.skip(trainSize)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



    print(f"Creating model...")
    model = create_model()

    history = model.fit(train_dataset, epochs=50, validation_data=val_dataset)

    with open('bronchoModelTrainingHistory.json', 'w') as f:
        json.dump(history.history, f)


    print(f"Saving model...")
    model.save("BronchoModel.keras")
    
    tf.saved_model.save(model, "broncho_model")
    




if __name__ == '__main__':
    train()
    
















