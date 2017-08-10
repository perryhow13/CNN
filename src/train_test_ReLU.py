from convLayer import convLayer
from ReLU import ReLU
from avgPoolLayer import avgPoolLayer
from fcLayer import fcLayer
from mnist_utils import cnn_load_MNIST
from object_utils import *
import numpy as np

# Load the data
X_train, y_train, X_test, y_test = cnn_load_MNIST()

# Set the learning rate
learning_rate = 0.00007

# Construct CNN 6c5-s2-12c5-s2-o
convLayer0 = convLayer(5,6,1,0,learning_rate)
relu0 = ReLU()
avgLayer0 = avgPoolLayer(2)
convLayer1 = convLayer(5,12,1,0,learning_rate)
relu1 = ReLU()
avgLayer1 = avgPoolLayer(2)
fcLayer0 = fcLayer(10, learning_rate)

# Create a file to log the accuracy
log_accuracy = open("log_ReLU_accuracy.txt", "w")

# Train with the whole training set
trainSet_size = 60000
testSet_size = 10000
epoch_size = 10

log_accuracy.write("Learning rate: " + str(learning_rate) + "\n")
log_accuracy.write("The size of training set:" + str(trainSet_size) + "\n")
log_accuracy.write("The size of testing set :" + str(testSet_size) + "\n")
log_accuracy.write("Epoch: " + str(epoch_size) + "\n")

for y in range(epoch_size):
  for z in range(trainSet_size):
    print epoch + str(y)
    print str(z+1) + "/" + str(trainSet_size)
    a = convLayer0.forward_pass_Conv(X_train[z])
    b = relu0.forward_ReLU(a)
    c = avgLayer0.forward_avgPool(b)
    d = convLayer1.forward_pass_Conv(c)
    e = relu1.forward_ReLU(d)
    f = avgLayer1.forward_avgPool(e)
    g = fcLayer0.forward_fcLayer(f)

    h = fcLayer0.backward_fcLayer(y_train[z])
    i = avgLayer1.backward_avgPool(h)
    j = relu1.backward_ReLU(i)
    k = convLayer1.backward_pass_Conv(j)
    l = avgLayer0.backward_avgPool(k)
    m = relu0.backward_ReLU(l)
    n = convLayer0.backward_pass_Conv(m)

  count = 0.0
  for i in range(testSet_size):
    a = convLayer0.forward_pass_Conv(X_test[i])
    b = relu0.forward_ReLU(a)
    c = avgLayer0.forward_avgPool(b)
    d = convLayer1.forward_pass_Conv(c)
    e = relu1.forward_ReLU(d)
    f = avgLayer1.forward_avgPool(e)
    g = fcLayer0.forward_fcLayer(f)
    
    if np.argmax(g) == y_test[i]:
      count += 1

  accuracy = (count/testSet_size) * 100
  log_accuracy.write("epoch " + str(y+1) + " accuracy : " + str(accuracy) + "% \n")

log_accuracy.close()

save_object(convLayer0, 'convLayer0.pkl')
save_object(avgLayer0, 'avgLayer0.pkl')
save_object(convLayer1, 'convLayer1.pkl')
save_object(avgLayer1, 'avgLayer1.pkl')
save_object(fcLayer0, 'fcLayer0.pkl')