import numpy as np
from mlp import MLP
from learning_rate_S import LeanringRateS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

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

    def __init__(self, layer_size, los, dropout_training, dropout_rates,  activation, Lr = 0.01, batch_size = 32, regularization='none', lambda_val=0.01, optimizer_type='adam', scheduler_type = 'step'):
        
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
        self.scheduler_type = scheduler_type
        self.optimizer_type = optimizer_type

        self.mlp = MLP(self.layer_size, 
                       self.dropout_training,
                        self.dropout_rates, 
                       self.activations,
                       self.Lr,
                       self.regularization, 
                       self.lambda_val, 
                       optimizer_type)
        
        if self.scheduler_type:
            self.scheduler = LeanringRateS(self.Lr, self.scheduler_type)
        else:
            self.scheduler = None

######### Regularaiation ########    
    def reg_loss(self, loss, reg_x):
        #self.reg_x = reg_x
        m = reg_x.shape[0]
        #calcualte the weights of each layer
        reg_w = 0.0
        for layer in self.mlp.layers:
            if self.regularization == 'L2':
                reg_w += np.sum(layer.w**2)
            elif self.regularization == 'L1':
                reg_w += np.sum(np.abs(layer.w))
   
        if self.regularization == 'L2':
            #explain the formula in the markdown
            reg_loss = loss + ((self.lambda_val / (2 * m)) * (reg_w))
        elif self.regularization == 'L1':
            reg_loss = loss + ((self.lambda_val / m) * (reg_w))
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

            #update lr
            if self.scheduler:
                self.Lr = self.scheduler.learning_rate_step(epoch)

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
                
                self.mlp.step()
                #self.mlp.step(gradiants, self.Lr)
            
            ### Rerunning the whole data on the network to check if it learned
            y_pred_full_train = self.mlp.forward(X_train, dropout_training = False)
            full_train_loss1 = 0.0
            if self.los == "MSE":
                full_train_loss1 = self.los_object.MSE(y_pred_full_train, y_train)
            elif self.los == "BCE":
                full_train_loss1 = self.los_object.binary_cross_entropy(y_pred_full_train, y_train)

            ### Adding the regularaization to the loss
            #full_train_loss = self.binary_cross_entropy(y_pred_full_train, y_train)
            #full_train_loss = self.reg_loss(full_train_loss1, X_train)
            full_train_loss = np.array(self.reg_loss(full_train_loss1, X_train)).item()
            full_train_acc = self.accuracy(y_pred_full_train, y_train)

            self.loss_history.append(full_train_loss)
            self.accuracy_history.append(full_train_acc)

            #validating the model larning on new
            if x_val is not None and y_val is not None:
                y_pred_val = self.mlp.forward(x_val, dropout_training = False)
                val_loss = 0.0
                if self.los == "MSE":
                    val_loss = self.los_object.MSE(y_pred_val, y_val)
                elif self.los == "BCI":
                    val_loss = self.los_object.binary_cross_entropy(y_pred_val, y_val)

                #val_loss = self.binary_cross_entropy(y_pred_val, y_val)
                #val_loss = self.reg_loss(val_loss, x_val)
                val_loss = np.array(self.reg_loss(val_loss, x_val)).item()
                val_acc = self.accuracy(y_pred_val, y_val)

                self.eval_loss_history.append(val_loss)
                self.eval_acc_history.append(val_acc)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                val_info = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%" if x_val is not None else ""
                print(f"Epoch {epoch:4d} | Loss: {full_train_loss:.4f} | Acc: {full_train_acc:.2f}%{val_info}")


    
    def accuracy(self, y_pred, y_true):
        if y_true.shape[1] == 1: # Binary (WDBC in this case)
            predictions = (y_pred >= 0.5).astype(int)
        else: # Multi-class (MNIST in his case)
            predictions = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == y_true_labels) * 100
    
        return np.mean(predictions == y_true) * 100
        #y_pred = (y_pred >= 0.5).astype(int)
        #return np.mean(y_pred == y_true) * 100


    def evaluate_test_set(self, X_test, y_test):
        """Computes Accuracy, Precision, Recall, F1, and Confusion Matrix"""
        # Get raw probabilities
        y_pred_probs = self.mlp.forward(X_test, dropout_training=False)
        
        # Handle Binary vs Multi-class labels
        if y_test.shape[1] == 1: # Binary (WDBC)
            y_pred_labels = (y_pred_probs >= 0.5).astype(int)
            y_true_labels = y_test
        else: # Multi-class (MNIST)
            y_pred_labels = np.argmax(y_pred_probs, axis=1)
            y_true_labels = np.argmax(y_test, axis=1)

        # Calculate Metrics
        metrics = {
            "Accuracy": self.accuracy(y_pred_probs, y_test),
            "Precision": precision_score(y_true_labels, y_pred_labels, average='macro'),
            "Recall": recall_score(y_true_labels, y_pred_labels, average='macro'),
            "F1 Score": f1_score(y_true_labels, y_pred_labels, average='macro'),
            "Confusion Matrix": confusion_matrix(y_true_labels, y_pred_labels)
        }
        return metrics, y_pred_probs, y_true_labels
    

    def predict(self, X):
        y_pred = self.mlp.forward(X, dropout_training=False)
        if y_pred.shape[1] == 1:
            return (y_pred >= 0.5).astype(int)
        return np.argmax(y_pred, axis=1)
        #return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.mlp.forward(X, dropout_training = False)
    

    def plot_learning_curves(self):
        """Plots Training and Validation progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss Curve
        ax1.plot(self.loss_history, label='Train Loss', color='blue')
        if self.eval_loss_history:
            ax1.plot(self.eval_loss_history, label='Val Loss', color='red')
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy Curve
        ax2.plot(self.accuracy_history, label='Train Acc', color='blue')
        if self.eval_acc_history:
            ax2.plot(self.eval_acc_history, label='Val Acc', color='red')
        ax2.set_title('Accuracy Curve')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, conf_matrix, labels=None):
        """Plots the heatmap of correct vs incorrect predictions"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_roc_curve(self, y_true, y_probs):
        """Plots ROC curve for binary classification (WDBC)"""
        if y_true.shape[1] == 1:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.show()
