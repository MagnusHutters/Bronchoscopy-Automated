from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the Keras model
model = load_model('BronchoModel.keras')

# Define the endpoint for inference
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json  # Assuming JSON data is sent

    # Preprocess the input image data
    image_data = preprocess_image(data)

    # Perform inference
    prediction = model.predict(np.array([image_data]))

    # Postprocess the prediction
    output = postprocess_prediction(prediction)

    # Return the result
    return jsonify(output)

# Function to preprocess input image data
def preprocess_image(data):
    # Implement your preprocessing logic here
    # Example: Convert JSON data to a suitable input format for the model
    # Example: Convert base64 encoded image to NumPy array
    return np.array(data['image'])

# Function to postprocess the model prediction
def postprocess_prediction(prediction):
    # Implement your postprocessing logic here
    # Example: Convert the model output to a format suitable for sending back to the Raspberry Pi
    return prediction.tolist()  # Convert to list for JSON serialization

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)  # Run on all available network interfaces
