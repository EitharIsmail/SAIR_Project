import numpy as np
from mlp import MLP

#building the classiifer
class loss_functions:
    def __init__(self, Y_true, y_pred):
        self.Y_true = Y_true
        self.y_pred = y_pred

    def MSE(self, Y_true, y_pred):
        L = np.mean((Y_true - y_pred) ** 2)
        return L

    def MSE_grad(self, Y_true, y_pred):
        n = Y_true.shape[0]
        L_grad = (2 /n) * (Y_true - y_pred)
        return L_grad

    def binary_cross_entropy(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #to avoid log(0)
        L = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return L
    
    #its gradiant
    def binary_cross_entropy_grad(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        L = (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))
        return L
    

class NeruronNetworkLearning:

    def __init__(self, layer_size, los, dropout_training, dropout_rates,  activation, Lr = 0.01, batch_size = 32, regularization='none', lambda_val=0.01):
        
        self.layer_size = layer_size
        self.activations = activation
        self.batch_size = batch_size
        self.Lr = Lr
        self.loss_history = []
        self.accuracy_history = []
        self.eval_loss_history = []
        self.eval_acc_history = []
        self.lambda_val = lambda_val 
        self.regularization = regularization
        self.los = los
        self.dropout_training = dropout_training
        self.dropout_rates = dropout_rates

        self.mlp = MLP(self.layer_size, 
                       self.dropout_training,
                        self.dropout_rates, 
                       self.activations,
                       self.Lr,
                       self.regularization, 
                       self.lambda_val, 
                       optimizer_type = 'adam')

######### Regularaiation ########    
    def reg_loss(self, loss, reg_x):
        self.reg_x = reg_x
        #calcualte the weights of each layer
        reg_w = 0.0
        for layer in self.mlp.layers:
            if self.regularization == 'L2':
                reg_w += np.sum(layer.w**2)
            elif self.regularization == 'L1':
                reg_w += np.sum(np.abs(layer.w))
   
        if self.regularization == 'L2':
            #explain the formula in the markdown
            reg_loss = loss + ((self.lambda_val / (2 * self.reg_x[0])) * (reg_w))
        elif self.regularization == 'L1':
            reg_loss = loss + ((self.lambda_val / self.reg_x[0]) * (reg_w))
        else:
            reg_loss = loss
        
        return reg_loss

########## Training the data by looping through epochs and dividing it to batches for initial training
    def train(self, X_train, y_train, x_val, y_val, epochs = 1000, verbose = True):
        self.X_train = X_train
        n_samples = X_train.shape[0]
        self.y_train = y_train
        self.los_object = loss_functions(self.y_train, self.mlp.layer.y_pred)

        for epoch in range(epochs):

             #to avoid learning one type of output, it will memoeries leading to overtting, also we didn't use suffle function because it will break the dataset sequance
            index = np.random.permutation(n_samples)
            x_shaffeled = X_train[index]
            y_shaffeled = y_train[index]

            #loop to create batchs and train them
            for start_idx in range(0, n_samples, self.batch_size):
                end_index = min(start_idx + self.batch_size, n_samples)
                #slizing the dataset based on the batch size
                x_batch = x_shaffeled[start_idx : end_index]
                y_batch = y_shaffeled[start_idx: end_index]

                y_predd = self.mlp.forward(x_batch, dropout_training = True)

                dL_dy = 0.0
                if self.los == "MSE":
                    dL_dy = self.los_object.MSE_grad(y_predd, y_batch)
                elif self.los == "BCE":
                    dL_dy = self.los_object.binary_cross_entropy_grad(y_predd, y_batch)
                #dL_dy = self.los_object.binary_cross_entropy_grad(y_predd, y_batch)

                gradiants = self.mlp.backward(dL_dy)
                
                self.mlp.step(gradiants, self.Lr)
            
            ### Rerunning the whole data on the network to check if it learned
            y_pred_full_train = self.mlp.forward(X_train, dropout_training = False)
            full_train_loss1 = 0.0
            if self.los == "MSE":
                full_train_loss1 = self.los_object.MSE(y_predd, y_train)
            elif self.los == "BCE":
                full_train_loss1 = self.los_object.binary_cross_entropy(y_predd, y_train)
            ### Adding the regularaization to the loss
            #full_train_loss = self.binary_cross_entropy(y_pred_full_train, y_train)
            full_train_loss = self.reg_loss(full_train_loss1, X_train)
            full_train_acc = self.accuracy(y_pred_full_train, y_train)

            self.loss_history.append(full_train_loss)
            self.accuracy_history.append(full_train_acc)

            #validating the model larning on new
            if x_val is not None and y_val is not None:
                y_pred_val = self.mlp.forward(x_val, dropout_training = False)
                val_loss = 0.0
                if self.los == "MSE":
                    val_loss = self.los_object.MSE(y_predd, y_train, x_val)
                elif self.los == "BCE":
                    val_loss = self.los_object.binary_cross_entropy(y_predd, y_train)
                #val_loss = self.binary_cross_entropy(y_pred_val, y_val)
                val_loss = self.reg_loss(val_loss, x_val)
                val_acc = self.accuracy(y_pred_val, y_val)

                self.eval_loss_history.append(val_loss)
                self.eval_acc_history.append(val_acc)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%" if x_val is not None else ""
                print(f"Epoch {epoch:4d} | Loss: {full_train_loss:.4f} | Acc: {full_train_acc:.2f}%{val_info}")


    
    def accuracy(self, y_pred, y_true):
        y_pred = (y_pred >= 0.5).astype(int)
        return np.mean(y_pred == y_true) * 100
    

    def predict(self, X):
        """Make predictions"""
        y_pred = self.mlp.forward(X, dropout_training = False)
        return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.mlp.forward(X, dropout_training = False)
