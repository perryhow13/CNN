import numpy as np
from ReLU import ReLU

class fcLayer:
  def __init__(self, num_Class, learning_rate):
    self.num_Class = num_Class
    self.init_W = 0
    self.learning_rate = learning_rate
    self.input_volume = None
    self.reshaped_input = None
    self.weights = None
    self.scores= None
    self.activation = ReLU() #Softplus()

  def forward_fcLayer(self, input_volume):

    self.input_volume = input_volume
    #(D*W*H,1)
    self.reshaped_input = np.reshape(input_volume, (len(input_volume)*len(input_volume[0])*len(input_volume[0]),1))
      
    #(10, D*W*H)
    if self.init_W == 0:
      #self.weights = np.random.uniform(-0.1, 0.1, size=(self.num_Class,len(input_volume)*len(input_volume[0])*len(input_volume[0])))
      self.weights = 0.01 * np.random.randn(self.num_Class, len(input_volume)*len(input_volume[0])*len(input_volume[0]))
      self.init_W = 1
    
    #(10,1)  
    scores = np.dot(self.weights, self.reshaped_input)
    self.scores = self.activation.forward_ReLU(scores)

    return self.scores
  
  def backward_fcLayer(self, label):
    # Target
    correct_scores = np.zeros((self.num_Class,1))
    correct_scores[label] = 1

    # Loss 
    Loss = np.sum(1/2.*np.power(self.scores - correct_scores,2))
    print "Loss: " + str(Loss) + "\n"
      
    # dLoss/dscores (10, 1)
    dscores = self.scores - correct_scores

    #dLoss/dweights (10,D*W*H)
    dweights = np.dot(self.activation.backward_ReLU(dscores), np.transpose(self.reshaped_input)) 

    # Update weights
    self.weights -= self.learning_rate*dweights
      
    # dLoss/dinput_volume (1,D*W*H)
    dreshaped_input = np.dot(np.transpose(self.activation.backward_ReLU(dscores)), self.weights) 

    # drepshapd_input which is shape of (D,W,H)
    dreshaped_input = dreshaped_input.reshape(len(self.input_volume), len(self.input_volume[0]), len(self.input_volume[0]))
      
    return dreshaped_input