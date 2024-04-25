import tensorflow as tf

def convert_to_tflite(path):
    model = tf.keras.models.load_model(path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional: Set more detailed conversion options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True  # Enable the new converter

    try:
        tflite_model = converter.convert()
        return tflite_model
    except Exception as e:
        print(f"Failed to convert model {path}: {e}")
        return None

# Convert and save the model
tflite_model = convert_to_tflite('model.keras')
if tflite_model:
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
        print("Model converted successfully!")