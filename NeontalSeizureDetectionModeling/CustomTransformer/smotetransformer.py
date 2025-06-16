# smotetransformer.py

import torch
import torch.nn as nn

class SMOTETransformer(nn.Module):
    def __init__(self, input_size=256, num_classes=2, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(SMOTETransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_size)
        """
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
