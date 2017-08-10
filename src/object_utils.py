import pickle

def save_object(object, file_name):
  with open(file_name, 'wb') as output:
    pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

def load_object(file_name):
  with open(file_name, 'rb') as input:
    obj = pickle.load(input)

  return obj