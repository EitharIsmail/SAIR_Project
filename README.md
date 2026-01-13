# SAIR_Project

🧠 Custom Neural Network Library (built from Scratch using Numpy)
This project is a modular, from-scratch implementation of a Multi-Layer Perceptron (MLP) using NumPy. It includes advanced features typically found in modern frameworks like PyTorch or TensorFlow, such as multiple activation functions, advanced optimizers (Adam, RMSProp), regularization, and learning rate scheduling.

🚀 How to Run
1. Prerequisites
Ensure you have Python installed (3.8+ recommended) along with the following libraries:

Bash

pip install numpy pandas matplotlib seaborn scikit-learn
2. File Structure
Ensure all project files are in the same directory (or properly path-referenced):

main.py: The entry point to run experiments.

model_train.py: Contains the NeruronNetworkLearning wrapper class.

mlp.py: Logic for stacking layers and handling forward/backward passes.

denslayer.py: The core Dense (Linear) layer implementation.

optimizer.py: Implementation of SGD, Momentum, RMSprop, and Adam.

learning_rate_S.py: Learning rate decay strategies.

model_init_real_data.py: Factory for dataset-specific configurations.

3. Executing the Project
To run the default experiment (Breast Cancer/WDBC dataset):

Bash

python main.py
To change datasets: Open main.py and modify the run_experiment call to use 'mnist' (ensure you have the MNIST data files in the specified path).

To adjust hyperparameters: Edit the configs dictionary in model_init_real_data.py.

🛠 Features
1. Architecture & Layers
Modular MLP: Allows for an arbitrary number of hidden layers and neurons.

Activation Functions: Supports ReLU, Sigmoid, Tanh, and Linear.

Dropout: Built-in dropout layers to prevent overfitting during training.

2. Optimization & Training
Advanced Optimizers: * SGD & Momentum

RMSprop

Adam (with bias correction).

Schedulers: Exponential and Step decay for the learning rate.

Loss Functions: Mean Squared Error (MSE) and Binary Cross-Entropy (BCE).

3. Regularization
L1 (Lasso) and L2 (Ridge) regularization are integrated into the weight update step to penalize large weights and improve generalization.

📊 Evaluation & Visualization
The library automatically generates performance reports, including:

Learning Curves: Plots of Training vs. Validation Loss and Accuracy.

Confusion Matrix: A heatmap showing true vs. predicted classifications.

Metrics: Detailed Precision, Recall, and F1-score.

ROC Curve: Specifically for binary classification tasks (like WDBC).

🔧 Troubleshooting
Data Paths: In main.py, ensure the wdbc_path or mnist_path correctly points to where your data is stored on your local machine.

Gradient Stability: If the loss becomes NaN, try reducing the Lr (Learning Rate) or ensuring your input data is standardized using the loader.standardize() method provided in the pipeline.R group
