# SAIR_Project

🧠 Custom Neural Network Library (built from Scratch using Numpy)
This project is a modular, from-scratch implementation of a Multi-Layer Perceptron (MLP) using NumPy. It includes advanced features typically found in modern frameworks like PyTorch or TensorFlow, such as multiple activation functions, advanced optimizers (Adam, RMSProp), regularization, and learning rate scheduling.

🚀 How to Run
1. Prerequisites
Ensure you have Python installed (3.8+ recommended) along with the following libraries:

Bash

pip install numpy pandas matplotlib seaborn scikit-learn


**2. File Structure**

Ensure all project files are in the same directory (or properly path-referenced):

main.py: The entry point to run experiments.

model_train.py: Contains the NeruronNetworkLearning wrapper class.

mlp.py: Logic for stacking layers and handling forward/backward passes.

denslayer.py: The core Dense (Linear) layer implementation.

optimizer.py: Implementation of SGD, Momentum, RMSprop, and Adam.

learning_rate_S.py: Learning rate decay strategies.

model_init_real_data.py: Factory for dataset-specific configurations.

**3. Executing the Project**

To run the default experiment (Breast Cancer/WDBC dataset):

Bash

python NeuronNetworkLibrary\src\main.py