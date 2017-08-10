import numpy as np

class ReLU:  
  def __init__(self):
    self.input_volume = None
        
  def forward_ReLU(self, input_volume):
    self.input_volume = input_volume
    output_volume = np.maximum(self.input_volume, 0)
    return output_volume
    
  def backward_ReLU(self, input_gradient):
    output_volume = input_gradient
    output_volume[self.input_volume <= 0] = 0
    return output_volume