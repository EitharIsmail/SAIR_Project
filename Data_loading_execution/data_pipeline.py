import numpy as np
import pandas as pd
import struct
import os

class DataLoader:

    @staticmethod
    def load_mnist(folder_path):
        """Loads MNIST binary files from a directory"""
        # We use the exact names seen in your screenshot
        img_path = os.path.join(folder_path, 'train-images.idx3-ubyte')
        lbl_path = os.path.join(folder_path, 'train-labels.idx1-ubyte')

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing MNIST images at: {img_path}")

        # 1. Read Images
        with open(img_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

        # 2. Read Labels
        with open(lbl_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)

        # 3. One-hot encode labels (0-9)
        num_classes = 10
        y_one_hot = np.eye(num_classes)[y]
        
        # Normalize to [0, 1] and ensure 2D
        return X.astype(float) / 255.0, y_one_hot

    # ... keep train_val_test_split ...
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