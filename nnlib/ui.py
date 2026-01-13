import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_loading_execution.data_pipeline import DataLoader
from model_init_real_data import ModelFactory
from model_train import NeruronNetworkLearning

# Global variable to store the trained model
trained_model_system = None

def start_training(dataset_name, epochs, lr, batch_size, progress=gr.Progress()):
    global trained_model_system
    
    # 1. Load Data
    loader = DataLoader()
    # Replace with your local paths
    path = f"data/{dataset_name}_data" 
    if dataset_name == 'wdbc':
        X, y = loader.load_wdbc(path)
    else:
        X, y = loader.load_mnist(path)
    
    X = loader.standardize(X)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.train_val_test_split(X, y)

    # 2. Get Config & Initialize
    config = ModelFactory.get_config(dataset_name)
    config['Lr'] = lr
    config['batch_size'] = batch_size
    
    # Custom training loop to yield plots for real-time visualization
    model = NeruronNetworkLearning(**config)
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        # Progress bar update
        progress(epoch / epochs, desc=f"Training Epoch {epoch}/{epochs}")
        
        # Train one epoch (Assuming your .train method can be run epoch-wise)
        # Note: You may need to expose an 'train_step' method in model_train.py
        model.train(X_train, y_train, X_val, y_val, epochs=1) 
        
        loss_history.append(model.loss_history[-1])
        
        # Create plot for real-time update
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history, label='Loss', color='blue')
        plt.title(f"Training Loss - Epoch {epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        yield plt, f"Training... Loss: {loss_history[-1]:.4f}"

    trained_model_system = model
    yield plt, "Training Complete! ✅"

def make_prediction(input_data):
    if trained_model_system is None:
        return "Please train a model first."
    
    # Convert input string to numpy array
    try:
        data = np.fromstring(input_data, sep=',').reshape(1, -1)
        # Forward pass
        probs = trained_model_system.mlp.forward(data, dropout_training=False)
        prediction = np.argmax(probs) if probs.shape[1] > 1 else (probs > 0.5).astype(int)
        return f"Prediction: {prediction} (Confidence: {np.max(probs):.2f})"
    except Exception as e:
        return f"Error: {str(e)}"

# UI Layout using Blocks
with gr.Blocks(title="Neural Network Dashboard") as demo:
    gr.Markdown("# 🧠 Neural Network Training Dashboard")
    
    with gr.Tab("Configuration & Training"):
        with gr.Row():
            dataset = gr.Dropdown(["wdbc", "mnist"], label="Select Dataset", value="wdbc")
            epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
        
        with gr.Row():
            lr = gr.Number(value=0.01, label="Learning Rate")
            batch_size = gr.Number(value=32, label="Batch Size")
            
        train_btn = gr.Button("🚀 Start Training", variant="primary")
        
        with gr.Row():
            plot_output = gr.Plot(label="Live Training Progress")
            status_text = gr.Textbox(label="Status")

    with gr.Tab("Prediction"):
        input_box = gr.Textbox(label="Input Features (comma-separated)", placeholder="0.1, 0.5, 1.2...")
        predict_btn = gr.Button("Predict")
        pred_output = gr.Label(label="Result")

    # Connect components
    train_btn.click(start_training, inputs=[dataset, epochs, lr, batch_size], outputs=[plot_output, status_text])
    predict_btn.click(make_prediction, inputs=[input_box], outputs=[pred_output])

if __name__ == "__main__":
    demo.launch()