import numpy as np

class Optimizers:
    def __init__(self, layers,optimizer, parameters_gradient,  beta1 = 0.9,  beta2= 0.999, lr = 0.01):
        self.layers = layers
        self.parameters_gradient = parameters_gradient
        self.optimizer = optimizer
        self.lr = lr
        self.beta1 = beta1 #for momentum
        self.beta2 = beta2 #for rms and ADAM optimizer
        self.upadated_parameters = []
        self.t = 0

        # initialize veloicity for each layer to match w and b dimentions
        self.w_velocity = []
        self.b_velocity = []

        for layer in self.layers:
            self.w_velocity.append(np.zeros_like(layer.w))
            self.b_velocity.append(np.zeros_like(layer.b))
        
        #cashe for rmsprop
        self.cache_w = []
        self.cache_b = []
        for layer in layers:
            self.cache_w.append(np.zeros_like(layer.w))
            self.cache_b.append(np.zeros_like(layer.b))

        #for adam initialize veloicy and cache
        self.adam_w = []
        self.adam_b = []

        for layer in self.layers:
            self.adam_w.append(np.zeros_like(layer.w))
            self.adam_b.append(np.zeros_like(layer.b))

        self.adam_w_cache = []
        self.adam_b_cache = []

        for layer in self.layers:
            self.adam_w_cache.append(np.zeros_like(layer.w))
            self.adam_b_cache.append(np.zeros_like(layer.b))
        
 
    def update_by_optimizers(self, optimizer, parameters_gradient):

        # 1. SGD : Stochastic Gradient Descent
        if optimizer == 'sgd':
            for i, (dL_dw, dL_db) in enumerate(parameters_gradient):
                self.layers[i].w -= self.lr * dL_dw
                self.layers[i].b -= self.lr * dL_db

                self.upadated_parameters.append((self.layers[i].w, self.layers[i].b))

        # 2. Momentum
        elif optimizer == 'momentum':
            for i, (dL_dw, dL_db) in enumerate(parameters_gradient):
                #calculate the velocities
                self.w_velocity[i] = self.beta1 * self.w_velocity[i] + (1 - self.beta1) * dL_dw
                self.b_velocity[i] = self.beta1 * self.b_velocity[i] + (1 - self.beta1) * dL_db # the momentum value

                #update the parameters
                self.layers[i].w -= self.lr * self.w_velocity[i] 
                self.layers[i].b -= self.lr * self.b_velocity[i]

                self.upadated_parameters.append((self.layers[i].w, self.layers[i].b))

        # 3. RMSprop:
        elif optimizer == 'rmsprop':
            #calculate the cashe
            for i, (dL_dw, dL_db) in enumerate(parameters_gradient):
                self.cache_w[i] = self.beta2 * self.cache_w[i] + (1 - self.beta2) * dL_dw ** 2
                self.cache_b[i] = self.beta2 * self.cache_b[i] + (1 - self.beta2) * dL_db ** 2
                
                #update parameter
                epsilon=1e-8  #to avoid division by zero
                self.layers[i].w -= self.lr * dL_dw / (np.sqrt(self.cache_w[i]) + epsilon)
                self.layers[i].b -= self.lr * dL_db / (np.sqrt(self.cache_b[i]) + epsilon)

                self.upadated_parameters.append((self.layers[i].w, self.layers[i].b))

        # 4. Adam:        
        elif optimizer == 'adam':
            epsilon=1e-8
            for i, (dL_dw, dL_db) in enumerate(parameters_gradient):
                # Momentum
                self.adam_w[i] = self.beta1 * self.adam_w[i] + (1 - self.beta1) * dL_dw
                self.adam_b[i] = self.beta1 * self.adam_b[i] + (1 - self.beta1) * dL_db

                #rmsprop
                self.adam_w_cache[i] = self.beta2 * self.adam_w_cache[i] + (1 - self.beta2) * dL_dw ** 2
                self.adam_b_cache[i] = self.beta2 * self.adam_b_cache[i] + (1 - self.beta2) * dL_db ** 2

                #Bias correction
                #velocity : Corrected first moment
                ad_v_w = self.adam_w[i] / (1 - self.beta1 **self.t)
                as_v_b = self.adam_b[i] / (1 - self.beta1 **self.t)

                #rmsprop : Corrected second moment
                ad_rms_w = self.adam_w_cache[i] / (1 - self.beta2 ** self.t)
                ad_rms_b = self.adam_b_cache[i] / (1 - self.beta2 ** self.t)

                #adam update
                self.layers[i].w -= self.lr * ad_v_w / (np.sqrt(ad_rms_w) + epsilon)
                self.layers[i].b -= self.lr * as_v_b / (np.sqrt(ad_rms_b) + epsilon)

                self.upadated_parameters.append((self.layers[i].w, self.layers[i].b))
            
        return self.upadated_parameters
    
    def optimizers_step(self):
        """Convenience method to perform one optimization step."""
        return self.update_by_optimizers(self.optimizer, self.parameters_gradient)
     

