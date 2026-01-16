import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Data_loading_execution.data_pipeline import DataLoader
from NeuronNetworkLibrary.src.model_train import NeruronNetworkLearning

# Global variable to store the trained model
trained_model_system = None

def start_training(dataset_name, epochs, lr, batch_size, progress=gr.Progress()):
    global trained_model_system
    
    # 1. Load Data
    loader = DataLoader()
    
    # Corrected paths based on your verified structure
    if dataset_name == 'wdbc':
            path = "Data/Breast_Cancer/wdbc.data" # Points to the file
            X, y = loader.load_wdbc(path)
    else:
            path = "Data/MNIST" # Points to the folder containing .idx files
            X, y = loader.load_mnist(path)
    
    X = loader.standardize(X)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.train_val_test_split(X, y)

    # 2. Manual Config (Since ModelFactory might vary)
    # Matching the params required by your NeruronNetworkLearning class
    config = {
        "layer_size": [X_train.shape[1], 16, 8, 1] if dataset_name == 'wdbc' else [784, 128, 64, 10],
        "los": "BCE" if dataset_name == 'wdbc' else "MSE",
        "dropout_training": False,
        "dropout_rates": [],
        "activation": "relu",
        "Lr": lr,
        "batch_size": int(batch_size),
        "regularization": 'none',
        "optimizer_type": 'adam',
        "scheduler_type": 'step'
    }
    
    model = NeruronNetworkLearning(**config)
    loss_history = []
    
    for epoch in range(1, int(epochs) + 1):
        # Progress bar update
        progress(epoch / epochs, desc=f"Training Epoch {epoch}/{epochs}")
        
        # Train one epoch
        model.train(X_train, y_train, X_val, y_val, epochs=1, verbose=False)
        
        loss_history.append(model.loss_history[-1])
        
        # Create plot for real-time update
        fig = plt.figure(figsize=(10, 4))
        plt.plot(loss_history, label='Train Loss', color='#2563eb') # Deep Blue
        plt.title(f"Training Loss - Epoch {epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        yield fig, f"Running Epoch {epoch}... Current Loss: {loss_history[-1]:.4f}"
        plt.close(fig)

    trained_model_system = model
    yield plt.gcf(), "Training Complete! ✅ Model is ready for prediction."

def make_prediction(input_data):
    if trained_model_system is None:
        return "Please train a model first."
    
    try:
        # Clean the input string and convert to float array
        data = np.array([float(x.strip()) for x in input_data.split(',')]).reshape(1, -1)
        
        # Standardize if necessary (Note: ideally use the loader's scaler)
        probs = trained_model_system.predict_proba(data)
        
        if probs.shape[1] > 1:
            prediction = np.argmax(probs)
            confidence = np.max(probs)
        else:
            prediction = 1 if probs >= 0.5 else 0
            confidence = probs[0][0] if prediction == 1 else 1 - probs[0][0]
            
        return {f"Class {prediction}": float(confidence)}
    except Exception as e:
        return f"Error: {str(e)}. Ensure you entered the correct number of features."

# UI Layout with Blue Theme
with gr.Blocks(theme=gr.themes.Ocean(), title="Neural Network Dashboard") as demo:
    gr.Markdown("# 🧠 Neural Network Training Dashboard")
    gr.Markdown("Configure your model and monitor training in real-time.")
    
    with gr.Tab("Configuration & Training"):
        with gr.Row():
            dataset = gr.Dropdown(["wdbc", "mnist"], label="Select Dataset", value="wdbc")
            epochs = gr.Slider(10, 1000, value=100, step=10, label="Epochs")
        
        with gr.Row():
            lr = gr.Number(value=0.01, label="Learning Rate")
            batch_size = gr.Number(value=32, label="Batch Size")
            
        train_btn = gr.Button("🚀 Start Training", variant="primary")
        
        with gr.Row():
            plot_output = gr.Plot(label="Live Training Progress")
            status_text = gr.Textbox(label="Status")

    with gr.Tab("Prediction"):
        gr.Markdown("### Enter feature values separated by commas")
        input_box = gr.Textbox(label="Input Features", placeholder="e.g. 17.99, 10.38, 122.8, 1001.0...")
        predict_btn = gr.Button("Predict", variant="primary")
        pred_output = gr.Label(label="Result")

    # Event Handlers
    train_btn.click(start_training, inputs=[dataset, epochs, lr, batch_size], outputs=[plot_output, status_text])
    predict_btn.click(make_prediction, inputs=[input_box], outputs=[pred_output])

if __name__ == "__main__":
    demo.launch()