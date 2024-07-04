

import numpy as np


class Input:
    actions = ['f', 'b', 'l', 'r', 'u', 'd']
    
    def __init__(self, int_value):
        self.int_value = int_value
    
    @classmethod
    def fromInt(cls, value):
        if 0 <= value <= 5:
            return cls(value)
        else:
            raise ValueError("Integer value must be between 0 and 5")
    
    @classmethod
    def fromChar(cls, char):
        if char in cls.actions:
            int_value = cls.actions.index(char)
            return cls(int_value)
        else:
            raise ValueError("Invalid character for action")
    
    @classmethod
    def fromModel(cls, values):
        try:
            values = np.array(values, dtype=float)
            if values.shape != (6,):
                raise ValueError
            int_value = int(np.argmax(values))
            return cls(int_value)
        except:
            raise ValueError("Values must be a list, tuple, or numpy array of six numerical values")
    
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
    
    def toInt(self):
        return self.int_value
    
    def toChar(self):
        return self.actions[self.int_value]
    
    def toModel(self):
        return [1.0 if i == self.int_value else 0.0 for i in range(6)]
    
    def toDict(self):
        return {
            "int_value": self.toInt(),
            "char_value": self.toChar(),
            "model_value": self.toModel()
        }