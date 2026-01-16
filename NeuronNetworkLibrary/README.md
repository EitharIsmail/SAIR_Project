# Explanation of the important classes in the neuron library

# 1. Neuron Class



# 2. DenseLayer:

This class is to Stack the neurons in one layer to create a layer


# 3. MLP:

This class is to create a full neuron network by stacking denselayers together.


# Important concepts applied to optimize the model:


**Dropout:**

Dropout is a regularization technique used in neural networks to prevent overfitting. During training, it randomly "drops out" (sets to zero) a fraction of neurons in a layer. This prevents neurons from becoming too dependent on specific features and encourages the network to learn more robust, generalized representations.

Key characteristics:

- Only applied during training, not during inference/prediction.

- Neurons are dropped randomly for each training sample/forward pass.

- Acts as an ensemble method - training multiple subnetworks simultaneously.

During implementation:

- Implemented During training where neurons are randomly dropped

- But! During inference all neurons are used, but outputs are scaled



**Optimizers:**

Optimizers are algorithms that adjust neural network weights to minimize the loss function. They determine how and how much to update weights based on gradients during training.

Analogy:

Loss function = Mountain you want to descend

Gradients = Slope/direction

Optimizer = How you walk down (small steps, big steps, momentum, etc.)

Types:

1. Stochastic Gradient Descent (SDG):

The simplest optimizer: w = w - learning_rate * gradient

2. Momentum:

Here step depends on weighted average of past gradients.

Like a rolling ball - has momentum, can't change direction instantly.

Smoother, faster convergence.

3. Root Mean Square Propagation (RMSprop):

is an adaptive learning rate algorithm that speeds up training by adjusting learning rates per parameter, using a moving average of squared gradients to normalize updates.

The Core Idea

Problem: Different parameters have different gradient magnitudes.

Solution: Scale learning rate inversely with gradient magnitude.

Intuition:

Large gradient → Divide by large number → Small step

Small gradient → Divide by small number → Large step

Result: All parameters move at similar rates!

4. ADAM

Mix the Momentum and the Root Mean Square Propagation (RMSprop)
