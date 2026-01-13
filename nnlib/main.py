import numpy as np
# Assuming your folder structure matches the provided path
from Data_loading_execution.data_pipeline import DataLoader
from model_init_real_data import ModelFactory

def run_experiment(dataset_name, file_path, epochs=100):
    print("\n" + "="*45)
    print(f"🚀 STARTING EXPERIMENT: {dataset_name.upper()}")
    print("="*45)
   
    # 1. Load and Preprocess using DataLoader
    loader = DataLoader()
    if dataset_name == 'wdbc':
        X, y = loader.load_wdbc(file_path)
    elif dataset_name == 'mnist':
        X, y = loader.load_mnist(file_path)
    else:
        raise ValueError("Unsupported dataset name.")

    # Apply standardization
    X = loader.standardize(X)
    
    # Split the data (70% Train, 15% Val, 15% Test)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.train_val_test_split(X, y)

    # 2. Initialize Model using the Factory
    model_trainer = ModelFactory.create_model(dataset_name)

    # 3. Train
    model_trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)

    # 4. FULL EVALUATION SYSTEM
    print(f"\n📊 Evaluating {dataset_name.upper()} on Test Set...")
    
    # Get predictions and compute all metrics (Precision, Recall, F1, CM)
    metrics, y_probs, y_true = model_trainer.evaluate_test_set(X_test, y_test)
    
    # Print numerical results
    for metric_name, value in metrics.items():
        if metric_name != "Confusion Matrix":
            print(f"🔹 {metric_name}: {value:.4f}" if "Accuracy" not in metric_name else f"🔹 {metric_name}: {value:.2f}%")

    # 5. VISUALIZATION
    # Loss and Accuracy curves
    model_trainer.plot_learning_curves()
    
    # Confusion Matrix Heatmap
    labels = ['Benign', 'Malignant'] if dataset_name == 'wdbc' else [str(i) for i in range(10)]
    model_trainer.plot_confusion_matrix(metrics["Confusion Matrix"], labels=labels)
    
    # ROC Curve (Only for Binary Classification like WDBC)
    if dataset_name == 'wdbc':
        model_trainer.plot_roc_curve(y_test, y_probs)

    print("="*45)

if __name__ == "__main__":
    # Path for WDBC
    wdbc_path = '3_Neural Network from scratch/nn_capstone_example/src/nnlib/Data/Breast_Cancer/wdbc.data'
    run_experiment(dataset_name='wdbc', file_path=wdbc_path, epochs=100)
    
# Path for MNIST - Point to the FOLDER containing the ubyte files
    #mnist_path = '3_Neural Network from scratch/nn_capstone_example/src/nnlib/Data/MNIST' 
    #run_experiment(dataset_name='mnist', file_path=mnist_path, epochs=10)