


import tflite_runtime.interpreter as tflite


class TFLiteModel:
    def __init__(self, model_path):
        # Initialize the TFLite interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *inputs):
        # Check if the number of inputs matches the expected number
        if len(inputs) != len(self.input_details):
            raise ValueError(f"Expected {len(self.input_details)} inputs, but got {len(inputs)}.")

        # Set the inputs to the model
        for i, input_data in enumerate(inputs):
            #print(input_data)
            if input_data.shape != tuple(self.input_details[i]['shape']):
                raise ValueError(f"Input {i} has incorrect shape. Expected {self.input_details[i]['shape']}, got {input_data.shape}.")
            self.interpreter.set_tensor(self.input_details[i]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Extract and return the output
        output_data = [self.interpreter.get_tensor(output['index']) for output in self.output_details]
        return output_data if len(output_data) > 1 else output_data[0]