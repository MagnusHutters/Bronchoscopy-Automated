

import numpy as np


class Input:
    actions = ['f', 'b', 'l', 'r', 'u', 'd']
    
    no_input_char = ''  # Character for no input
    no_input_int = -1   # Integer value for no input
    
    def __init__(self, int_value=0):
        
        
        if not (-1 <= int_value <= 5):
            raise ValueError("Integer value must be between -1 and 5, inclusive")
        self.int_value = int_value
        
        
    
    @classmethod
    def fromInt(cls, value):
        return cls(value)
    
    
    @classmethod
    def fromChar(cls, char):
        if char == cls.no_input_char:
            return cls(cls.no_input_int)
        if char in cls.actions:
            int_value = cls.actions.index(char)
            return cls(int_value)
        else:
            raise ValueError("Invalid character for action")
    
    @classmethod
    def fromModel(cls, values):
        values = np.array(values, dtype=float)
        if values.shape != (6,):
            raise ValueError("Model values must be an array of six numerical values")
        if np.all(values == 0):  # Check if all values are exactly zero, indicating no input
            return cls(cls.no_input_int)
        max_value = np.max(values)
        indices = [i for i, v in enumerate(values) if v == max_value]
        # Select the first index with the maximum value, handling ties by default order
        return cls(indices[0])
    
    
    @classmethod
    def fromDict(cls, dict_values):
        int_value = dict_values.get("int_value")
        if int_value is not None:
            return cls.fromInt(int_value)
        char_value = dict_values.get("char_value")
        if char_value is not None:
            return cls.fromChar(char_value)
        model_value = dict_values.get("model_value")
        if model_value is not None:
            return cls.fromModel(model_value)
        raise ValueError("Dictionary must contain 'int_value', 'char_value', or 'model_value'")
    
    @classmethod
    def fromJoystick(cls, joystick):
        # Round joystick values to nearest 0.01 to avoid near-zero noise
        forwards = round(joystick.forwards, 2)
        rotation = round(joystick.rotation, 2)
        bend = -round(joystick.bend, 2)
        
        #print(f"Joystick: {forwards}, {rotation}, {bend}")

        model_values = [0] * 6  # Corresponds to 'f', 'b', 'l', 'r', 'u', 'd'
        
        # Map rounded joystick inputs to actions
        if forwards > 0:
            model_values[0] = forwards  # 'f'
        elif forwards < 0:
            model_values[1] = -forwards  # 'b'

        if rotation > 0:
            model_values[3] = rotation  # 'r'
        elif rotation < 0:
            model_values[2] = -rotation  # 'l'

        if bend > 0:
            model_values[4] = bend  # 'u'
        elif bend < 0:
            model_values[5] = -bend  # 'd'

        # Check for no input state
        if all(v == 0 for v in model_values):
            return cls(cls.no_input_int)

        # Determine the action with the maximum value (prioritize lower index on tie)
        max_value = max(model_values)
        int_value = model_values.index(max_value)  # Returns the index of the first occurrence of max value
        return cls(int_value)
    
    def toInt(self):
        return self.int_value
    
    def toChar(self):
        if self.int_value == self.no_input_int:
            return self.no_input_char
        return self.actions[self.int_value]
    
    def toModel(self):
        return [1.0 if i == self.int_value else 0.0 for i in range(6)]
    
    def toDict(self):
        return {
            "int_value": self.toInt(),
            "char_value": self.toChar(),
            "model_value": self.toModel()
        }
        
        
    #string representation of the object
    def __str__(self):
        return self.toChar()