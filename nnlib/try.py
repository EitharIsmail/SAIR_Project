import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from model_train import NeruronNetworkLearning

# Generate datasets
print("📊 GENERATING DATASETS...")
np.random.seed(42)

# Moons dataset
X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)
y_moons = y_moons.reshape(-1, 1)

# Circles dataset  
X_circles, y_circles = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
y_circles = y_circles.reshape(-1, 1)

print(f"Moons dataset: X {X_moons.shape}, y {y_moons.shape}")
print(f"Circles dataset: X {X_circles.shape}, y {y_circles.shape}")

# Split into train/validation sets
X_moons_train, X_moons_val, y_moons_train, y_moons_val = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42
)

X_circles_train, X_circles_val, y_circles_train, y_circles_val = train_test_split(
    X_circles, y_circles, test_size=0.2, random_state=42
)

print(f"\nTrain/Validation splits:")
print(f"Moons - Train: {X_moons_train.shape}, Val: {X_moons_val.shape}")
print(f"Circles - Train: {X_circles_train.shape}, Val: {X_circles_val.shape}")


print("\n" + "="*60)
print("🌙 TRAINING ON MOONS DATASET")
print("="*60)

# Create and train neural network for moons
nn_moons = NeruronNetworkLearning(
    layer_size=[2, 16, 8, 1],  # Input: 2, Hidden: 16→8, Output: 1
    los = 'MSE',
    dropout_training= True,
    dropout_rates = [0.1, 0.0, 0.02],
    activation=['Relu', 'Relu', 'Sigmoid'],
    Lr=0.1,
    batch_size=32,
    scheduler_type='exponential'
)

#print("🧠 Neural Network Architecture for Moons:")
#for i, layer in enumerate(nn_moons.mlp.layers):
#    print(f"Layer {i}: {layer.w.shape[0]} → {layer.w.shape[1]} neurons ({layer.activation})")

# Train the model
nn_moons.train(
    X_moons_train, y_moons_train,
    x_val=X_moons_val, y_val=y_moons_val,
    epochs=1000,
    verbose=True
)

# Final evaluation
final_pred_moons = nn_moons.predict(X_moons_val)
final_proba_moons = nn_moons.predict_proba(X_moons_val)
final_acc_moons = nn_moons.accuracy(final_proba_moons, y_moons_val)

print(f"\n🎯 Final Moons Validation Accuracy: {final_acc_moons:.2f}%")