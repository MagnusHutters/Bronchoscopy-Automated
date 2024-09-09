

import torch

#version
print(torch.__version__)

#check if cuda is available
print(torch.cuda.is_available())

#device info
print(torch.cuda.get_device_name(0))
