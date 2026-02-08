# main_app/apps.py

from django.apps import AppConfig

# -------------------------------
# ❌ ML IMPORTS DISABLED (Render Free Tier)
# -------------------------------
# import os
# import pickle
# import torch
# import torch.nn as nn

# -------------------------------
# ❌ ML MODEL CLASS DISABLED
# -------------------------------
# class HybridRNNModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
#                  n_layers, bidirectional, dropout, tabular_feature_count, padding_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
#         self.lstm = nn.LSTM(
#             embedding_dim, hidden_dim,
#             num_layers=n_layers,
#             bidirectional=bidirectional,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.tabular_fc = nn.Linear(tabular_feature_count, 32)
#         self.relu = nn.ReLU()
#         lstm_output_size = hidden_dim * 2
#         self.fc_combined = nn.Linear(lstm_output_size + 32, output_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text, tabular_features):
#         embedded = self.embedding(text)
#         _, (hidden, _) = self.lstm(embedded)
#         hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
#         tabular_out = self.relu(self.tabular_fc(tabular_features))
#         combined = torch.cat((hidden, tabular_out), dim=1)
#         return self.fc_combined(combined)


# -------------------------------
# ✅ SAFE APP CONFIG (NO ML LOADING)
# -------------------------------
class MainAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main_app'

    # ❌ ML GLOBALS DISABLED
    # MODEL = None
    # VOCAB = None
    # TRAIN_COLS = None
    # DEVICE = None

    # ❌ ready() METHOD DISABLED (CAUSES OOM)
    # def ready(self):
    #     pass
