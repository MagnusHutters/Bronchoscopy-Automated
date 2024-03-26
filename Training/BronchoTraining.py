

import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate



from EpisodeLoader import *



def imageLoader(image_path):
    tf.print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0  # Convert img to float32 and normalize
    return img

def create_model():
    image_input_shape = (256, 256, 3)  # Assuming RGB images
    pathInputShape = (4, 4)
    statesInputShape = (1,2)
    #predictionsInputShape = (4, 3)




    # Define the inputs
    image_input = Input(shape=image_input_shape, name='image_input')
    pathInput = Input(shape=pathInputShape, name='pathInput')
    statesInput = Input(shape=statesInputShape, name='statesInput')
    #predictionsInput = Input(shape=statesInputShape, name='predictionsInput')

    # Define the CNN model for image input
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flattenCNN = Flatten()(pool2)

    #flatten first
    pathDense = Flatten()(pathInput)
    statesDense = Flatten()(statesInput)

    # Define Dense layers for other inputs
    pathDense = Dense(16, activation='relu')(pathDense)
    statesDense = Dense(4, activation='relu')(statesDense)
    #predictionsDense = Dense(16, activation='relu')(predictionsInput)
    #flatten the dense layers
    pathDense = Flatten()(pathDense)
    statesDense = Flatten()(statesDense)
    #predictionsDense = Flatten()(predictionsDense)


    concatenatedArrays = concatenate([pathDense, statesDense])
    #another dense
    dense_array1 = Dense(64, activation='relu')(concatenatedArrays)

    # Concatenate all inputs
    concatenated_inputs = concatenate([flattenCNN, dense_array1])

    # Dense layer to combine features
    combined_features = Dense(128, activation='relu')(concatenated_inputs)

    # Dense layer to combine features
    combined_features = Dense(64, activation='relu')(combined_features)

    # Output layer - output is a 3-element vector representing output in range -1 to 1
    output = Dense(3, activation='linear')(combined_features)

    # Define the model
    model = tf.keras.Model(inputs=[image_input, pathInput, statesInput], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()


    return model








def train():


    #load data
    path = "Training\Data\BronchoData"

    episodes = os.listdir(path)

    images = []
    inputs = []
    states = []
    paths = []
    #predictions = []

    for episode in episodes:
        episodePath = os.path.join(path, episode)
        episodeImages, episodeInputs, episodeStates, episodePaths, episodePredictions = loadEpisode(episodePath)
        images.extend(episodeImages)
        inputs.extend(episodeInputs)
        states.extend(episodeStates)
        paths.extend(episodePaths)
        #predictions.extend(episodePredictions)

    imageDataset = tf.data.Dataset.from_tensor_slices(images)
    stateDataset = tf.data.Dataset.from_tensor_slices(states)
    pathDataset = tf.data.Dataset.from_tensor_slices(paths)
    labelInput = tf.data.Dataset.from_tensor_slices(inputs)

    combined_dataset = tf.data.Dataset.zip((imageDataset, pathDataset, stateDataset, labelInput))

    combined_dataset = combined_dataset.map(
        lambda img_path, arr1, arr2, label: (
            (imageLoader(img_path), arr1, arr2), label
            )
    )

    buffer_size = 1000  # This is just an example; adjust based on your dataset size and memory constraints
    batch_size = 32

    combined_dataset = combined_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)


    model = create_model()

    model.fit(combined_dataset, epochs=10)

    model.save("BronchoModel.keras")




if __name__ == '__main__':
    train()
    
















