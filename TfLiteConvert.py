import os
import sys
import tensorflow as tf

def convert_models_to_tflite(model_paths):
    for path in model_paths:
        if not path.endswith('.keras'):
            print(f"Skipping {path}: File does not have a .keras extension")
            continue
        
        model = tf.keras.models.load_model(path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        base = os.path.splitext(path)[0]
        tflite_path = base + '.tflite'
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Converted and saved {tflite_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path1> <path2> ... <pathN>")
        sys.exit(1)
    
    model_paths = sys.argv[1:]
    convert_models_to_tflite(model_paths)
