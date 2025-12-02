import torch
import torch.nn as nn


# Aggiornamento per Colab - v2
class NeuroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(NeuroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layer LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Layer Finale di Classificazione
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Inizializza stati nascosti a zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Prendi l'output dell'ultimo step temporale
        out = out[:, -1, :]
        out = self.fc(out)
        return out