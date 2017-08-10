import numpy as np

class avgPoolLayer:
  def __init__(self, filter_Size):
    self.filter_Size = filter_Size
    self.stride = filter_Size
    self.input_volume = None
      
  def forward_avgPool(self, input_volume):    
    self.input_volume = input_volume

    # Check if the output size is integer or not
    assert (len(input_volume[0]) - self.filter_Size) % self.stride == 0

    output_size = (len(input_volume[0]) - self.filter_Size)/self.stride + 1
    output_volume = np.empty([len(input_volume), output_size, output_size], dtype=float)

    for i in range(len(input_volume)):
      for j in range(output_size):
        for k in range(output_size):
          output_volume[i][j,k] = (1.0/self.filter_Size**2)*np.sum(input_volume[i][j*self.stride:j*self.stride+self.filter_Size,k*self.stride:k*self.stride+self.filter_Size])

    return output_volume
  
  def backward_avgPool(self, input_gradient):
      
    assert len(input_gradient) == len(self.input_volume)
      
    output_size = len(self.input_volume[0])
    output_volume = np.empty([len(self.input_volume), output_size, output_size], dtype=float)
      
    for i in range(len(input_gradient)):
      for j in range(len(input_gradient[0])):
        for k in range(len(input_gradient[0])):
          output_volume[i][j*self.stride:j*self.stride+self.filter_Size,k*self.stride:k*self.stride+self.filter_Size] = (1.0/self.filter_Size**2)*input_gradient[i][j,k]
   
    return output_volume