import numpy as np

class convLayer:
  def __init__(self, filter_Size, num_Filters, stride, pad, learning_rate):
    self.filter_Size = filter_Size
    self.num_Filters = num_Filters
    self.stride = stride
    self.pad = pad
    self.init_W = 0
    self.weights = None
    self.input_volume = None
    self.learing_rate = learning_rate

  def forward_pass_Conv(self, input_volume):       
    self.input_volume = input_volume
      
    if self.init_W == 0:
      #self.weights = np.random.uniform(-0.1, 0.1, size=(self.num_Filters, len(input_volume), self.filter_Size, self.filter_Size))
      self.weights = 0.01 * np.random.randn(self.num_Filters, len(input_volume), self.filter_Size, self.filter_Size)
      self.init_W = 1
          
    assert (len(input_volume[0]) - self.filter_Size + 2*self.pad) % self.stride == 0
      
    output_size = (len(input_volume[0]) - self.filter_Size + 2*self.pad)/self.stride + 1
    output_volume = np.empty([self.num_Filters, output_size, output_size], dtype = float)
      
    if self.pad != 0:
      img = np.empty([len(input_volume), len(input_volume[0])+2*self.pad, len(input_volume[0])+2*self.pad], dtype=float)
      for i in range(len(input_volume)):
        img[i] = np.lib.pad(input_volume[i], self.pad, 'constant').tolist()
    else:
      img = input_volume
          
    for i in range(self.num_Filters):
      for j in range(output_size):
        for k in range(output_size):
          output_volume[i][j,k] = np.sum(self.weights[i] * img[:, j*self.stride:j*self.stride+self.filter_Size, k*self.stride:k*self.stride+self.filter_Size])
    
    return output_volume
  
  def backward_pass_Conv(self, input_gradient):
      
    # derivative of weights
    dweights = np.zeros(self.weights.shape, dtype = float)
    
    for i in range(self.num_Filters):
      for l in range(len(self.input_volume)):
        for j in range(self.filter_Size):
          for k in range(self.filter_Size):
            dweights[i][l][j,k] = np.sum(input_gradient[i] * self.input_volume[l][j*self.stride:j*self.stride+len(input_gradient[0]),k*self.stride:k*self.stride+len(input_gradient[0])])

    self.weights -= self.learing_rate*dweights
    
    # derivative of input_volume
  
    # rotate all the weights 180 degrees
    rotated_weights = np.empty(self.weights.shape, dtype = float)
    
    for i in range(len(self.weights)):
      for j in range(len(self.weights[0])):
        rotated_weights[i][j] = np.rot90(self.weights[i][j],2)
    
    assert ((len(self.input_volume[0])-1)*self.stride - len(input_gradient[0]) + self.filter_Size) % 2 == 0
    pad_size = ((len(self.input_volume[0])-1)*self.stride - len(input_gradient[0]) + self.filter_Size)/2
    
    if pad_size != 0:
      padded_input_gradient = np.empty((len(input_gradient), len(input_gradient[0])+2*pad_size, len(input_gradient[0])+2*pad_size), dtype = float)
      for i in range(len(input_gradient)):
        padded_input_gradient[i] = np.lib.pad(input_gradient[i], pad_size, 'constant').tolist()
    else:
      padded_input_gradient = input_gradient
                        
    assert (len(padded_input_gradient[0]) - self.filter_Size) % self.stride == 0
    output_size = (len(padded_input_gradient[0]) - self.filter_Size)/self.stride + 1
    
    assert output_size == len(self.input_volume[0])
    output_volume = np.empty([len(self.input_volume), output_size, output_size], dtype = float)
    
    for i in range(len(self.input_volume)):
      for j in range(output_size):
        for k in range(output_size):
          temp_sum = 0
          for l in range(self.num_Filters):
            temp_sum += np.sum(rotated_weights[l][i] * padded_input_gradient[l][j*self.stride:j*self.stride+self.filter_Size, k*self.stride:k*self.stride+self.filter_Size])
                
          output_volume[i][j,k] = temp_sum   

    return output_volume