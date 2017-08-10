import pyNN.nest as sim
import numpy as np
import math
import random

def convLayer_Connect(cnn_convLayer):

  input_size = len(cnn_convLayer.input_volume[0])
  output_size = ((input_size - cnn_convLayer.filter_Size + 2*cnn_convLayer.pad)/cnn_convLayer.stride) + 1
  weights = cnn_convLayer.weights

  exci_conn_list = []
  inhi_conn_list = []
  
  for i in range(cnn_convLayer.num_Filters):
    for j in range(output_size):
      for k in range(output_size):
        out_index = j*output_size + k + i*(output_size**2)
        for l in range(len(cnn_convLayer.input_volume)):
          for m in range(cnn_convLayer.filter_Size):
            for n in range(cnn_convLayer.filter_Size):
              #If something goes wrong have a look at this
              in_index = (m + cnn_convLayer.stride*j)*input_size + (n + cnn_convLayer.stride*k) + l*(input_size**2)
              if weights[i][l,m,n] > 0:
                exci_conn_list.append((in_index, out_index, weights[i][l,m,n], 1.))
              elif weights[i][l,m,n] < 0:
                inhi_conn_list.append((in_index, out_index, weights[i][l,m,n], 1.))
  
  return exci_conn_list, inhi_conn_list

def poolLayer_Connect(cnn_poolLayer):

  input_size = len(cnn_poolLayer.input_volume[0])
  output_size = ((input_size - cnn_poolLayer.filter_Size)/cnn_poolLayer.stride) + 1
  
  conn_list = []

  for i in range(len(cnn_poolLayer.input_volume)):
    for j in range(output_size):
      for k in range(output_size):
        out_index = j*output_size + k + i*(output_size**2)
        for m in range(cnn_poolLayer.filter_Size):
          for n in range(cnn_poolLayer.filter_Size):
            in_index = (m + cnn_poolLayer.stride*j)*input_size + (n + cnn_poolLayer.stride*k) + i*(input_size**2)
            conn_list.append((in_index, out_index, 1.0/cnn_poolLayer.filter_Size**2, 1.))

  return conn_list

def fcLayer_Connect(cnn_fcLayer, num_label):

  input_size = len(cnn_fcLayer.reshaped_input)
  output_size = num_label
  weights  =  cnn_fcLayer.weights
  exci_conn_list = []
  inhi_conn_list = []

  for i in range(output_size):
    for j in range(input_size):
      if weights[i][j] > 0:
        exci_conn_list.append((j, i, weights[i][j], 1.))
      elif weights[i][j] < 0:
        inhi_conn_list.append((j, i, weights[i][j], 1.))

  return exci_conn_list, inhi_conn_list

def convLayer_Proj(pre_pop, cnn_convLayer, cell_params):

  output_depth = cnn_convLayer.num_Filters
  output_size = (len(cnn_convLayer.input_volume[0]) - cnn_convLayer.filter_Size + 2*cnn_convLayer.pad)/cnn_convLayer.stride + 1
  post_pop = sim.Population(np.prod((output_depth, output_size, output_size)), sim.IF_curr_exp, cell_params)
  exci_conn, inhi_conn = convLayer_Connect(cnn_convLayer)

  if len(exci_conn) > 0:
    sim.Projection(pre_pop, post_pop, sim.FromListConnector(exci_conn), target='excitatory')

  if len(inhi_conn) > 0:
    sim.Projection(pre_pop, post_pop, sim.FromListConnector(inhi_conn), target='inhibitory')

  return post_pop

def poolLayer_Proj(pre_pop, cnn_poolLayer, cell_params):
  
  conn = poolLayer_Connect(cnn_poolLayer)
  output_depth = len(cnn_poolLayer.input_volume)
  output_size = (len(cnn_poolLayer.input_volume[0]) - cnn_poolLayer.filter_Size)/cnn_poolLayer.stride + 1
  post_pop = sim.Population(np.prod((output_depth, output_size, output_size)), sim.IF_curr_exp, cell_params)

  if len(conn) > 0:
    sim.Projection(pre_pop, post_pop, sim.FromListConnector(conn), target='excitatory')

  return post_pop

def fcLayer_Proj(pre_pop, cnn_fcLayer, cell_params, num_label):
  exci_conn, inhi_conn = fcLayer_Connect(cnn_fcLayer, num_label)
  post_pop = sim.Population(num_label, sim.IF_curr_exp, cell_params)

  if len(exci_conn) > 0:
    sim.Projection(pre_pop, post_pop, sim.FromListConnector(exci_conn), target='excitatory')

  if len(inhi_conn) > 0:
    sim.Projection(pre_pop, post_pop, sim.FromListConnector(inhi_conn), target='inhibitory')

  return post_pop

def nextTime(rateParameter):
  return -math.log(1.0 - random.random()) / rateParameter

 # 1000. to make it ms (millisecond)
def poisson_generator(fire_rate, t_start, t_stop):
  poisson_train = []
  if fire_rate > 0:
    next_isi = nextTime(fire_rate)*1000.
    last_time = next_isi + t_start
    while last_time  < t_stop:
      poisson_train.append(last_time)
      next_isi = nextTime(fire_rate)*1000.
      last_time += next_isi
  return poisson_train

def mnist_poisson_gen(img_set, total_rate, dur, silence):
  if total_rate > 0:
    for i in range(len(img_set)):
      img_set[i] = (img_set[i]/sum(img_set[i])) * total_rate

  spike_src_data = [[] for i in range(len(img_set[0]))]

  for i in range(len(img_set)):
    start_time = i*(dur+silence)
    end_time = start_time + dur
    for j in range(len(img_set[0])):
      spikes = poisson_generator(img_set[i][j], start_time, end_time)
      if spikes != []:
        spike_src_data[j].extend(spikes)
  
  return spike_src_data

# img_set is (n,28*28) where n is the number of images
def inputLayer_Pop(img_set, total_rate = 0, dur = 1000, silence = 200):
  input_pop = sim.Population(len(img_set[0]), sim.SpikeSourceArray, {'spike_times' : []})
  spike_source_data = mnist_poisson_gen(img_set, total_rate, dur, silence)

  for i in range(len(img_set[0])):
    input_pop[i].spike_times = spike_source_data[i]
  
  return input_pop