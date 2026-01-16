import argparse
import yaml
import pickle
import numpy as np
import os
from Data_loading_execution.data_pipeline import DataLoader
from NeuronNetworkLibrary.src.model_train import NeruronNetworkLearning
from NeuronNetworkLibrary.src.evaluator import Evaluator

def run_training(config_path):
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    dataset_name = list(full_config.keys())[0]
    config = full_config[dataset_name]
    
    # 1. Load Data
    loader = DataLoader()
    data_path = config.get('data_path')
    X, y = loader.load_wdbc(data_path) if dataset_name == 'wdbc' else loader.load_mnist(data_path)
    X = loader.standardize(X)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.train_val_test_split(X, y)

    # 2. Initialize Model (Using parameters from your config.yaml)
    # Note: Ensure the keys in your YAML match these config.get() names
    model_trainer = NeruronNetworkLearning(
        layer_size=config.get('layer_size', [30, 16, 8, 1]),
        los=config.get('loss_type', 'BCE'),
        dropout_training=config.get('dropout_training', False),
        dropout_rates=config.get('dropout_rates', []),
        activation=config.get('activation', 'relu'),
        Lr=config.get('learning_rate', 0.01),
        batch_size=config.get('batch_size', 32),
        regularization=config.get('regularization', 'none'),
        lambda_val=config.get('lambda_val', 0.01),
        optimizer_type=config.get('optimizer', 'adam'),
        scheduler_type=config.get('scheduler', 'step')
    )
    
    # 3. Train
    model_trainer.train(X_train, y_train, X_val, y_val, epochs=config.get('epochs', 100))
    evaluator = Evaluator(model_trainer)
    evaluator.evaluate(X_test, y_test)
    evaluator.plot_history()
    evaluator.plot_confusion_matrix(X_test, y_test)
        
    # 4. Save
    model_filename = f"{dataset_name}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model_trainer, f)
    print(f"✅ Model saved as {model_filename}")

def main():
    parser = argparse.ArgumentParser(description="Neural Network CLI Framework")
    subparsers = parser.add_subparsers(dest="command")

    # --- TRAIN COMMAND ---
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True, help="Path to config.yaml")

    # --- PREDICT COMMAND ---
    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("--model", required=True, help="Path to trained .pkl model")
    pred_parser.add_argument("--input", required=True, help="Path to .npy file or comma-separated string")

    args = parser.parse_args()

    if args.command == "train":
        run_training(args.config)
    
    elif args.command == "predict":
        # Load the saved model
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        
        # Load input (assuming .npy for now)
        data = np.load(args.input)
        prediction = model.predict(data)
        print(f"🔮 Prediction Result: {prediction}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()