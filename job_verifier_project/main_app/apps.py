# main_app/apps.py

from django.apps import AppConfig
import os
import pickle
import torch
import torch.nn as nn

# --- The HybridRNNModel class definition remains the same ---
class HybridRNNModel(nn.Module):
    # ... (your model class code here, no changes needed)
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, tabular_feature_count, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.tabular_fc = nn.Linear(tabular_feature_count, 32)
        self.relu = nn.ReLU()
        lstm_output_size = hidden_dim * 2
        self.fc_combined = nn.Linear(lstm_output_size + 32, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, tabular_features):
        embedded = self.embedding(text)
        _, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        tabular_out = self.relu(self.tabular_fc(tabular_features))
        combined = torch.cat((hidden, tabular_out), dim=1)
        return self.fc_combined(combined)


# --- THIS IS THE CORRECTED PART ---
class MainAppConfig(AppConfig): # <- RENAMED CLASS
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main_app' # <- CORRECT NAME

    # --- Global variables to hold the loaded model and artifacts ---
    MODEL = None
    VOCAB = None
    TRAIN_COLS = None
    DEVICE = None

    def ready(self):
        if os.environ.get('RUN_MAIN'):
            print("Loading ML model and artifacts...")

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # --- The path to your "Models" folder in the root, not "ml_models" ---
            model_dir = os.path.join(base_dir, 'Models') # <- CORRECTED PATH
            
            model_save_path = os.path.join(model_dir, 'hybrid_model_glove_v2.pth')
            vocab_save_path = os.path.join(model_dir, 'vocab_glove_v2.pth')
            cols_save_path = os.path.join(model_dir, 'train_cols_glove_v2.pkl')
            
            # --- The rest of your model loading logic remains the same ---
            MainAppConfig.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            MainAppConfig.VOCAB = torch.load(vocab_save_path)

            with open(cols_save_path, 'rb') as f:
                MainAppConfig.TRAIN_COLS = pickle.load(f)

            tabular_feature_count = len([c for c in MainAppConfig.TRAIN_COLS if c not in ['text']])
            padding_idx = MainAppConfig.VOCAB['<pad>']

            EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM = 100, 256, 1
            N_LAYERS, BIDIRECTIONAL, DROPUT = 2, True, 0.5
            
            model = HybridRNNModel(
                len(MainAppConfig.VOCAB), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
                N_LAYERS, BIDIRECTIONAL, DROPUT, tabular_feature_count, padding_idx
            ).to(MainAppConfig.DEVICE)
            
            model.load_state_dict(torch.load(model_save_path, map_location=MainAppConfig.DEVICE))
            model.eval()
            
            MainAppConfig.MODEL = model
            print("ML Model loaded successfully!")