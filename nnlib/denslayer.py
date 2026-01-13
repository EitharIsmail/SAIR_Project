import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class DesnsLayer:
#adding the w initialization
    #n_output mean the number of neurons per layer
    def __init__(self, n_inputs, n_outputs, los, dropout_training, dropout_rate,activation, Lr = 0.01, regularization='none',lambda_val=0.01):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        #self.w = np.random.randn(n_inputs, n_outputs) * 0.1 # here we multiplied by 0.1 to avoid the explding of the gradients because if the values of w came large then z will be big
        self.b = np.zeros((1, n_outputs))
        self.Lr = Lr
        self.los = los 
        self.lose =[] #for viz
        self.lambda_val = lambda_val 
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        self.dropout_masks = [] #to store for back
        self.dropout_training = dropout_training
        self.y_pred = 0
        self.mask = None
        
        self.w = self.initialize_parameters(self.n_inputs, self.n_outputs, self.activation)

    def initialize_parameters(self, n_inputs, n_outputs, activation):
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.scale = 0.0

        if self.activation in ('Sigmoid', 'Tanh'):
            self.scale = np.sqrt(2.0 / (self.n_inputs + self.n_outputs))
                
        elif self.activation == 'Relu':
            self.scale = np.sqrt(2.0 / self.n_inputs)

        weight = np.random.randn(self.n_inputs, n_outputs) * self.scale * 0.1

        return weight
    
    
    #activation functions
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-np.clip(X, -250, 250)))
    
    def relu(self, X):
        return np.maximum(0, X)
    
 
    def tanh(self, X):
        return np.tanh(X)

   #the dropout_training is to inforce the condition of applying dropout during training only
    def forward(self, X, dropout_training=True):
        self.X = X
        self.z = X @ self.w + self.b
        #print(z)

        #calling activation functions to activate the neuron to produce output(prediction)
        if self.activation == 'Linear':
            self.y_pred = self.z
        elif self.activation == 'Relu':
            self.y_pred = self.relu(self.z)
        elif self.activation == 'Tanh':
            self.y_pred = self.tanh(self.z)
        elif self.activation == 'Sigmoid':
            self.y_pred = self.sigmoid(self.z)

        #return self.y_pred
        if self.dropout_training:
            if self.dropout_rate is not None and  self.dropout_rate > 0.0:
                #the condition is to convert the mask to bolean values then use the float to make it numeric values
                self.mask = (np.random.rand(*self.y_pred.shape) > self.dropout_rate).astype(float)
                #scale the mask value to makeup the dropped neurons.
                self.mask = self.mask / (1 - self.dropout_rate)
                self.y_pred = self.y_pred * self.mask

                self.dropout_masks.append(self.mask)
            else:
                mask = None
                self.dropout_masks.append(self.mask)
        
        return self.y_pred
    

    def loss(self, los, y_pred, Y_true):
        if los == "MSE":
            L = np.mean((Y_true - y_pred) ** 2)
            self.lose.append(L)
            n = Y_true.shape[0]
            return (2 /n) * (Y_true - y_pred)   #dL_dy
        elif los == "BCE":
            # use the clip
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #to avoid log(0)
            L = - np.mean(Y_true * np.log(y_pred) + (1 - Y_true) * np.log(1 - y_pred))
            self.lose.append(L)
            return (y_pred - Y_true) / Y_true.shape[0]
            #(y_pred * (1- y_pred) * len(Y_true))
        
        return 0    


    # backword --> calculate gradients
    def backword(self, dL_dy):

        # activations derivatieves
        #dy_dz = 0
        if self.activation == "Linear":
            dy_dz = 1
        elif self.activation == "Relu":
            dy_dz = (self.z > 0).astype(float)
        elif self.activation == "Tanh":
            dy_dz = 1 - self.y_pred**2
        elif self.activation == "Sigmoid":
            dy_dz = self.y_pred * (1 - self.y_pred)
        
        #apply dropout
        if self.mask is not None:
            dL_dy = dL_dy * self.mask

        dL_dz = dL_dy * dy_dz
        dL_dw = self.X.T @ dL_dz
        dL_db = np.sum(dL_dz, axis=0, keepdims=1) * 1
        dL_dx = dL_dz @ self.w.T

         # Add regularization gradient to enhance dl_dw to avoid overfitting / the gradient of the summation and abs is the sign
        if self.regularization == 'L2':
            dL_dw = dL_dw + ((self.lambda_val / self.X.shape[0]) * self.w)
        elif self.regularization == 'L1':
            dL_dw = dL_dw + ((self.lambda_val / self.X.shape[0]) * np.sign(self.w))
        else:  #double check if this necessury
            dL_dw
            

        return dL_dw, dL_db, dL_dx

    #Time to update w and b     
    def step(self, dL_dw, dL_db):
        #update parameters
        self.w = self.w - self.Lr * dL_dw
        self.b = self.b - self.Lr * dL_db

        return self.w, self.b
    
    def viz(self):
        plt.figure(figsize=(10, 6))
        plt.title("🧠 Neural Layer Architecture", fontsize=14, pad=20)

        # Draw neurons
        for i in range(self.n_inputs):  # Input layer
            plt.scatter(0, i, s=500, c='lightblue', edgecolors='black', zorder=5)
            plt.text(0, i, f'x{i+1}', ha='center', va='center', fontweight='bold')
            
        for j in range(self.n_outputs):  # Output layer
            plt.scatter(2, j, s=500, c='lightgreen', edgecolors='black', zorder=5)
            plt.text(2, j, f'a{j+1}', ha='center', va='center', fontweight='bold')
            
            # Draw connections
            for i in range(self.n_inputs):
                plt.plot([0, 2], [i, j], 'gray', alpha=0.3)

        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 4.5)
        plt.axis('off')
        plt.text(0, -0.3, f"Input Layer {self.n_inputs}, neurons", ha='center', fontsize=12)
        plt.text(2, -0.3, f"Dense Layer {self.n_outputs}, neurons", ha='center', fontsize=12)
        plt.show()

