from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def cnn_load_MNIST():# Load the data
  mndata = MNIST('/home/young/MNIST-CNN/MNIST')
  x_train, y_train  = mndata.load_training()
  x_test, y_test = mndata.load_testing()

  # Normalise the data 
  x_train = np.double(x_train)/255.
  y_train = np.array(y_train)
  x_test = np.double(x_test)/255.
  y_test = np.array(y_test)
  X_train = []
  X_test = []

  for i in range(len(x_train)):
    temp = x_train[i].reshape(28,28)
    X_train.append(temp[np.newaxis,:,:])

  X_train = np.array(X_train)
      
  for i in range(len(x_test)):
    temp = x_test[i].reshape(28,28)
    X_test.append(temp[np.newaxis,:,:])

  X_test = np.array(X_test)

  return X_train, y_train, X_test, y_test

def snn_load_MNIST(hz = 200):
  mndata = MNIST('/home/young/MNIST-CNN/MNIST')
  x_train, y_train = mndata.load_testing()
  x_test, y_test = mndata.load_testing()
  x_train = (np.double(x_train)/255.) * hz
  y_train = np.array(y_train)
  x_test = (np.double(x_test)/255.) * hz
  y_test = np.array(y_test)

  return x_train, y_train, x_test, y_test


#plot a MNIST digit
def plot_MNIST(img):
    plt.figure(figsize=(5,5))
    im = plt.imshow(np.reshape(img,(28,28)), cmap=cm.gray_r, interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()