import pyNN.nest as sim
from snn import *
from mnist_utils import snn_load_MNIST
from object_utils import load_object
import random

# Load the data
x_train, y_train, x_test, y_test = snn_load_MNIST()

# 
tau_syn = 5.

cell_params_lif = {'cm': 0.25,      #nF
                   'i_offset': 0.1, #nA
                   'tau_m': 20.0,   #ms
                   'tau_refrac': 1.,#ms
                   'tau_syn_E': tau_syn,#ms
                   'tau_syn_I': tau_syn,#ms
                   'v_reset': -65.0,#mV
                   'v_rest': -65.0, #mV
                   'v_thresh': -50.0#mV
                   }   

# Load the pickles
convLayer0 = load_object('convLayer0.pkl')
poolLayer0 = load_object('avgLayer0.pkl')
convLayer1 = load_object('convLayer1.pkl')
poolLayer1 = load_object('avgLayer1.pkl')
fcLayer0 = load_object('fcLayer0.pkl')

num_img = 10000
num_label = 10
duration = 1000
silence = 200

# need to run whole snn
sim.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)

random.seed(0)

input_result = inputLayer_Pop(x_test[0:num_img])
conv0_result = convLayer_Proj(input_result, convLayer0, cell_params_lif)
pool0_result = poolLayer_Proj(conv0_result, poolLayer0, cell_params_lif)

conv1_result = convLayer_Proj(pool0_result, convLayer1, cell_params_lif)

pool1_result = poolLayer_Proj(conv1_result, poolLayer1, cell_params_lif)
result = fcLayer_Proj(pool1_result, fcLayer0, cell_params_lif, num_label)

# Record the populations before run the simulation
input_result.record()
conv0_result.record()
pool0_result.record()
conv1_result.record()
pool1_result.record()
result.record()

# Run simulation
sim.run((duration+silence)*num_img)
spike_result = result.getSpikes()
result_spike_count = result.get_spike_counts()
sim.end()


count = np.zeros((num_label, num_img))

for i in range(len(spike_result )):
  time_index = int((spike_result [i,1]-1)/(duration+silence))
  count[int(spike_result [i,0]), time_index] += 1

num_correct = 0

for i in range(num_img):
  if y_test[i] == np.argmax(count[:,i]):
    num_correct += 1

print count

print "Accuracy: " + str((num_correct/float(num_img))*100) + "%"


#print conv0_result_spike_count
