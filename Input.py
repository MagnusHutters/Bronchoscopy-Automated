

import numpy as np


class Input:
    
    
    
    
    numStepsDefault = [
        4, #Bend
        16, #Rotation
        12  #Extension
    ] 
    
    letterToAxis = {
        'd': 0,
        'u': 0,
        'l': 1,
        'r': 1,
        'f': 2,
        'b': 2,
    }
    letterToChange = {
        'd': numStepsDefault[0],
        'u': -numStepsDefault[0],
        'r': numStepsDefault[1],
        'l': -numStepsDefault[1],
        'f': numStepsDefault[2],
        'b': -numStepsDefault[2],
    }
    
    actionToName = {
        -1: "No Input",
        0: "Down",
        1: "Up",
        2: "Right",
        3: "Left",
        4: "Forward",
        5: "Backward",
    }
    
    
    
    
    
    def __init__(self, axis=None, change=0):
        self.axis = axis # None, 0, 1, 2 - None (no input), bend, rotation, extension
        self.change = change # -127 to 127 number of steps to move. 0 is no movement, positive is forward, negative is backward
        
        #false if axis is None, true otherwise
        self.hasInput = axis is not None
        
        
        
    @classmethod
    def fromInt(cls, axis, change):
        return cls(axis, change)
    
    @classmethod
    def fromChar(cls, char):
        #if empty char, return no input
        if char == '':
            return cls(None, 0)
        
        if char in cls.letterToAxis:
            axis = cls.letterToAxis[char]
            change = cls.letterToChange[char]
            return cls(axis, change)
        else:
            raise ValueError("Invalid character for action")
    
    @classmethod
    def fromModel(cls, values):
        values = np.array(values, dtype=float)
        if values.shape != (6,):
            raise ValueError("Model values must be an array of six numerical values")
        if np.all(values == 0):  # Check if all values are exactly zero, indicating no input
            return cls(None, 0)
        max_value = np.max(values)
        indices = [i for i, v in enumerate(values) if v == max_value]
        # Select the first index with the maximum value, handling ties by default order
        
        #take steps from magnitude of max value, minimum of 1
        steps = max(1, int(abs(max_value)))
        return cls(indices[0], steps)
    
    
    @classmethod
    def fromDict(cls, dict_values):
        axis = dict_values.get("axis")
        change = dict_values.get("change")
        if axis is not None and change is not None:
            return cls.fromInt(axis, change)
        char_value = dict_values.get("char_value")
        if char_value is not None:
            return cls.fromChar(char_value)
        model_value = dict_values.get("model_value")
        if model_value is not None:
            return cls.fromModel(model_value)
        raise ValueError("Dictionary must contain 'axis', 'change', or 'model_value'")
    
    
    @classmethod
    def fromJoystick(cls, joystick):
        # Round joystick values to nearest 0.01 to avoid near-zero noise
        bend = -round(joystick.bend, 2)
        forwards = round(joystick.forwards, 2)
        rotation = round(joystick.rotation, 2)
        
        #if no input, return no input
        if forwards == 0 and rotation == 0 and bend == 0:
            return cls(None, 0)
        
        #get axis and change from max absoulte value
        
        absValues = [abs(bend), abs(rotation), abs(forwards)]
        axis = np.argmax(absValues)
        
        
        #number of steps from default number of steps, negative if backwards
        change = cls.numStepsDefault[axis] * np.sign([bend, rotation, forwards][axis])
        
        
        return cls(axis, change)
    
    
    def toInt(self):
        return self.axis, self.change
    
    
    def toActionValue(self):
        if self.axis is None:
            return -1
        return self.axis * 2 + (0 if self.change > 0 else 1)
    
    def toChar(self):
        sign = np.sign(self.change)
        axis = self.axis
        
        if axis is None:
            return ''
        if axis == 0:
            return 'u' if sign > 0 else 'd'
        if axis == 1:
            return 'r' if sign > 0 else 'l'
        if axis == 2:
            return 'f' if sign > 0 else 'b'
        
        
    def toModel(self):
        if self.axis is None or self.change == 0:
            return [0] * 6
        
        
        values = [0] * 6
        sign = np.sign(self.change)
        sign = 1 if sign == 0 else 0
        axis = self.axis*2
        
        activeChannel = axis + sign
        
        values[activeChannel] = abs(self.change)
        return values
    
    def toDict(self):
        return {
            "axis": self.axis,
            "change": self.change,
            "char_value": self.toChar(),
            "model_value": self.toModel()
        }
        
        
        
    def __repr__(self):
        action = self.toActionValue()
        
        return f"Action {action}: {self.actionToName[action]}, Axis: {self.axis}, Change: {self.change}"
    
    
    
    
        
        
        
        
        