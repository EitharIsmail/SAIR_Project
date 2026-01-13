import numpy as np
import pandas as pd

class DataLoader:
    @staticmethod
    def load_wdbc(file_path):
        
        # Columns: ID, Diagnosis (M/B), then 30 real-valued features
        data = pd.read_csv(file_path, header=None)
        y = data.iloc[:, 1].values  # Diagnosis is the 2nd column
        X = data.iloc[:, 2:].values  # Features start from 3rd column
        
        # One-hot encode labels (M -> [1, 0], B -> [0, 1])
        y_encoded = np.where(y == 'M', 1, 0).reshape(-1, 1)
        return X, y_encoded

    @staticmethod
    def load_mnist(file_path):
      
        data = pd.read_csv(file_path)
        y = data.iloc[:, 0].values.reshape(-1, 1)
        X = data.iloc[:, 1:].values
        
        # Image flattening is implicit here as CSV rows are already 784 pixels
        # One-hot encoding for 10 classes (0-9)
        num_classes = 10
        y_one_hot = np.eye(num_classes)[y.flatten()]
        return X, y_one_hot

    @staticmethod
    def standardize(X):
       
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    @staticmethod
    def train_val_test_split(X, y, train_size=0.7, val_size=0.15):
        
        n = X.shape[0]
        indices = np.random.permutation(n)
        
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]