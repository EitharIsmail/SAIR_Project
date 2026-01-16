


## 🖥️ User Interface & Demo

The project includes a web-based dashboard built with **Streamlit** that allows you to configure hyperparameters and watch the training process in real-time.

### How to Run the UI
1. Install dependencies: `pip install streamlit matplotlib seaborn sklearn`
2. Launch the app: `streamlit run ui.py`


## Project Structure
neural_network_project/
├── configs/
│   └── config.yaml          # Hyperparameters
├── data/                    # Dataset storage
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── neuron.py
│    ├── denslayer.py              
│   ├── mlp.py               
│   ├── model_train.py
│   ├── model_init_real_data.py
│   
│   ├── optimizer.py
│   └── data_pipeline.py
│
├── pyproject.toml
├── setup.py
├── requirements.txt
└── README.md