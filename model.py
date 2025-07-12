import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_model(nn.Module):
    """Recurrent Neural Network Model for Symptom Analysis"""
    
    def __init__(self, input_size=5000, hidden_size=128, output_size=24):
        super(RNN_model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))
