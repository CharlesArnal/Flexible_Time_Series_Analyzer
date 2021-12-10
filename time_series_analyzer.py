import numpy as np
import pandas as pd
import random

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow import keras

# Example of use at the bottom of this file

def modelEstimate(trainingFilename):
  """
    Fits a model using historical data.
    trainingFilename (str): path to training data (has to be readable by panda.read_csv)
    Returns a scikit-learn model selected using crossvalidation randomized search and 
    trained on the data
  """
  training_data = pd.read_csv(trainingFilename) 
  
  batch_size=10000
  n_steps = min(1000,round(len(training_data)/2))
  # shape [batch_size,n_steps,2]
  train, train_indices =  input_seqs_from_data(training_data,n_steps,batch_size)
  # shape [batch_size,n_steps]
  train_seq2seq1D_labels = returns_labels_from_data(training_data,train_indices,n_steps)

  
  param_grid = [
  {'n_rec_layers':[1,2],'add_long_and_mid_term_info':[True,False],'n_hid_dense_layers': [0,1,2],'width_layers':[5,10,20,30]},
   ]

  model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn= build_model,
                                        input_dim = np.shape(train)[-1], epochs = 5)

  print("Model selection via crossvalidation randomized search underway")

  rand_search = RandomizedSearchCV(model, param_grid, n_iter = 6, cv=3,scoring='neg_mean_squared_error',return_train_score=True,)
  rand_search.fit(train, train_seq2seq1D_labels)

  cvres = rand_search.cv_results_
  for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
  print("\n Best parameters :")
  print(rand_search.best_params_)

  return rand_search.best_estimator_

def modelForecast(testFilename, parameters):
  """
  Makes predictions on a test set using a provided model
  trainingFilename (str): path to test data (has to be readable by panda.read_csv)
  parameters : a trained scikit-learn model
  """
  model = parameters
  test_data = pd.read_csv(testFilename) 
  test, _ = input_seqs_from_data(test_data,len(test_data["yprice"]),1,rand_sample=False)
  predictions = model.predict(test)
  return predictions




def build_model(input_dim=2, n_rec_layers=2,n_hid_dense_layers=1,width_layers=10,add_long_and_mid_term_info=True):
    """
    Returns a (yet untrained) sequence to sequence convolutional recurrent neural network.
    The model takes as input arrays of the shape [batch_size,sequence_length,input_dim],
    and outputs arrays of the shape [batch_size, sequence_length]
    The number of recurrent layers and hidden dense layers, as well as their widths, can be specified as arguments to build_model
    One can also choose whether to add longer term information (using additional convolutional layers)
    """
    input_shape = keras.layers.Input(shape=[None,input_dim])
    input_shape = keras.layers.BatchNormalization()(input_shape)
  
    short_term_info = keras.layers.Conv1D(filters=width_layers, kernel_size=2, strides=1, padding="causal")(input_shape)
    for i in range(0,n_rec_layers-1):
      short_term_info = keras.layers.GRU(width_layers, return_sequences=True)(short_term_info)
    short_term_info = keras.layers.GRU(width_layers, return_sequences=True)(short_term_info)

    mid_term_info = keras.layers.Conv1D(filters=width_layers, kernel_size=10, strides=1, padding="causal")(input_shape)
    for i in range(0,n_rec_layers-1):
      mid_term_info = keras.layers.GRU(width_layers, return_sequences=True)(mid_term_info)
    mid_term_info = keras.layers.GRU(width_layers, return_sequences=True)(mid_term_info)

    long_term_info = keras.layers.Conv1D(filters=width_layers, kernel_size=100, strides=1, padding="causal")(input_shape)
    for i in range(0,n_rec_layers-1):
      long_term_info = keras.layers.GRU(width_layers, return_sequences=True)(long_term_info)
    long_term_info = keras.layers.GRU(width_layers, return_sequences=True)(long_term_info)

    if add_long_and_mid_term_info:
      merged = keras.layers.TimeDistributed(tf.keras.layers.Concatenate(axis=1))([short_term_info, mid_term_info,long_term_info])
    else:
      merged = keras.layers.TimeDistributed(tf.keras.layers.Concatenate(axis=1))([short_term_info])
    merged = keras.layers.TimeDistributed(keras.layers.Flatten())(merged)

    for i in range(0,n_hid_dense_layers):
      merged = keras.layers.TimeDistributed(keras.layers.Dense(width_layers))(merged)
    out = keras.layers.TimeDistributed(keras.layers.Dense(1))(merged)
    model = keras.Model(input_shape, out)

    model.compile(loss= "mse", optimizer="adam")
    return model



def input_seqs_from_data(data,length,N, rand_sample = True):
  """
  If rand_sample == True, randomly samples N subsequences of the given length from the provided dataframe,
  Returns a [N,length,2] array, as well as the starting index of each sampled sequence
  If rand_sample == False, returns a single sequence of the given length starting at index 0
  as a [1,length,2] array
  """
  list_of_seqs = []
  y = np.array(data["yprice"])
  x = np.array(data["xprice"])
  if rand_sample:
    L = len(y)
    indices = random.sample(range(0,L-length),N)
    for i in indices:
        list_of_seqs.append([list(my_tuple) for my_tuple in zip(y[i:i+length],x[i:i+length])])
    return np.array(list_of_seqs), indices
  else:
    list_of_seqs.append([list(my_tuple) for my_tuple in zip(y[:length],x[:length])])
    return np.array(list_of_seqs), [0]

def returns_labels_from_data(data,indices,length):
  """
  Takes as input the data (as a dataframe), the starting indices of some subsequences
  and their length (the same for all)
  Outputs the sequences of returns corresponding to these subsequences
  as a [len(indices),length] array
  """
  seq2seq1D_labels = []
  returns = np.array(data["returns"])
  for i in indices:
    seq2seq1D_labels.append([returns[i+j] for j in range(0,length)])
  return np.array(seq2seq1D_labels)


# Example of use
# test.csv and train.csv are expected to have three columns : "xprice", "yprice" and "returns"
# The goal is to predict returns using xprice and yprice
#-------------------------
test_data = pd.read_csv('test.csv') 
test, test_indices = input_seqs_from_data(test_data,len(test_data["yprice"]),1,rand_sample=False)
test_seq2seq1D_labels = returns_labels_from_data(test_data,test_indices,len(test_data["yprice"]))

my_model = modelEstimate('train.csv')
predictions = modelForecast('test.csv',my_model)

print("Mean squared error : ")
print(mean_squared_error(predictions,np.squeeze(test_seq2seq1D_labels)))
#-------------------------
