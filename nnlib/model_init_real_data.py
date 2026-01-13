from model_train import NeruronNetworkLearning

class ModelFactory:
    @staticmethod
    def get_config(dataset_name):
        configs = {
            'wdbc': {
                'layer_size': [30, 16, 8, 1],      # 30 inputs for WDBC features
                'los': 'BCE',                      # Binary Cross Entropy
                'activation': ['Relu', 'Relu', 'Sigmoid'],
                'Lr': 0.01,
                'batch_size': 32,
                'optimizer_type': 'adam',
                'scheduler_type': 'exponential',
                'dropout_training': True,          # Enable dropout for regularization
                'dropout_rates': [0.1, 0.1, 0.0],  # Small dropout for hidden layers
                'regularization': 'L2',            # Helps prevent overfitting on small tabular data
                'lambda_val': 0.01
            },
            'mnist': {
                'layer_size': [784, 128, 64, 10],  # 784 inputs for flattened pixels
                'los': 'CCE',                      # Categorical Cross Entropy
                'activation': ['Relu', 'Relu', 'Softmax'],
                'Lr': 0.001,
                'batch_size': 64,
                'optimizer_type': 'adam',
                'scheduler_type': 'step',
                'dropout_training': True,
                'dropout_rates': [0.2, 0.2, 0.0],  # Higher dropout for complex image patterns
                'regularization': 'none',
                'lambda_val': 0.0
            }
        }
        return configs.get(dataset_name)

    @staticmethod
    def create_model(dataset_name):
        config = ModelFactory.get_config(dataset_name)
        if config:
            return NeruronNetworkLearning(**config)
        raise ValueError(f"Dataset {dataset_name} configuration not found.")