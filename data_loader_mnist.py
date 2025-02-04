"""
data_loader_mnist.py
- Load and preprocess MNIST data (http://yann.lecun.com/exdb/mnist/)
"""

# Necessary packages
import numpy as np
import pandas as pd
from keras.datasets import mnist

def load_mnist_data(label_data_rate):
  """MNIST data loading.
  
  Args:
    - label_data_rate: ratio of labeled data
  
  Returns:
    - x_label, y_label: labeled dataset
    - x_unlab: unlabeled dataset
    - x_test, y_test: test dataset
  """
  # Import mnist data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  """
  Comment ine 28-33, for experiments with 85-15% train-test split.
  """
  x = np.concatenate((x_train, x_test))
  y = np.concatenate((y_train, y_test))

  from sklearn.model_selection import train_test_split
  test_size = 0.2
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

  # One hot encoding for the labels
  y_train = np.asarray(pd.get_dummies(y_train))
  y_test = np.asarray(pd.get_dummies(y_test))

  # Normalize features
  x_train = x_train / 255.0
  x_test = x_test / 255.0
    
  # Treat MNIST data as tabular data with 784 features
  # Shape
  no, dim_x, dim_y = np.shape(x_train)
  test_no, _, _ = np.shape(x_test)
  
  x_train = np.reshape(x_train, [no, dim_x * dim_y])
  x_test = np.reshape(x_test, [test_no, dim_x * dim_y])
  
  # Divide labeled and unlabeled data
  idx = np.random.permutation(len(y_train))
  
  # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
  label_idx = idx[:int(len(idx)*label_data_rate)]
  unlab_idx = idx[int(len(idx)*label_data_rate):]
  
  # Unlabeled data
  x_unlab = x_train[unlab_idx, :]
  
  # Labeled data
  x_label = x_train[label_idx, :]  
  y_label = y_train[label_idx, :]
  
  return x_label, y_label, x_unlab, x_test, y_test