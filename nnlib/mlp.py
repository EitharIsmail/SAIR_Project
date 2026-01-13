import numpy as np 
from denslayer import DesnsLayer
from optimizer import Optimizers

#model building stack of layers MLP
class MLP:
#dropout_rate=0.0
    def __init__(self, n_neuorns_per_layer, dropout_training, dropout_rates, list_activations, Lr = 0.01,  regularization='none', lambda_val=0.01, optimizer_type = 'adam'):
        #item in zero index is the intput it self
        self.n_neuorns_per_layer = n_neuorns_per_layer
        self.list_activations = list_activations or ["Relu"] * (len(n_neuorns_per_layer) - 2) + ["Sigmoid"] #as the variable n_neuorns_per_layer indicate the number of layers
        #to stack layers in it
        self.layers = [] 
        self.lose = []
        self.lambda_val = lambda_val 
        self.regularization = regularization
        self.Lr = Lr
        self.dropout_training = dropout_training
        self.dropout_rates = dropout_rates or [0.0] * (len(n_neuorns_per_layer) - 1)
        self.optimizer_type = optimizer_type
        los = "MSE"
        
        
        #for the first layer is X it self and it is not counted in the loop as a layer, I see it as a layer but the program doesn't, it see the second layer
        # we said - 1 to skip first layer as it is the input
        for i in range(len(n_neuorns_per_layer) - 1):
            #to match mask of the current layer
            layer_dropout_rate =  self.dropout_rates[i]
            self.layer = DesnsLayer(self.n_neuorns_per_layer[i], self.n_neuorns_per_layer[i + 1], los, layer_dropout_rate, self.list_activations[i], self.Lr, self.regularization, self.lambda_val)

            #creating a list of objects to call it later to access the class functions
            self.layers.append(self.layer)

        self.optimizer = Optimizers(self.layers, 
                                    parameters_gradient=[], 
                                    beta2=0.999,
                                    beta1=0.9, 
                                    optimizer='adam', 
                                    lr=self.Lr)


    def forward(self, X, dropout_training = True):
        #to save the input to the memeory to know it later
        self.X = X 

        #because for each layer its input is the output of previous layer
        self.previous_output = X 
        for layer in self.layers:
            self.previous_output = layer.forward(self.previous_output, dropout_training = True)
        #return the last output as the inner ones are not important to me
        return self.previous_output  
    
    def backward(self, dL_dy):
        self.parameters_gradient = []
        y_pred = self.previous_output

        current_grad = dL_dy
        for layer in reversed(self.layers):
            dL_dw, dL_db, dL_dx = layer.backword(current_grad)
            self.parameters_gradient.append((dL_dw, dL_db))
            current_grad = dL_dx

            self.parameters_gradient = list(reversed(self.parameters_gradient))
            self.optimizer.parameters_gradient = self.parameters_gradient

        return self.parameters_gradient

    
    def step(self):
        updated_parameter = self.optimizer.optimizers_step()
        return updated_parameter

            
