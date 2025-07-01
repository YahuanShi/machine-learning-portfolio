import torch
import torch.nn as nn

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3, output_dim_reg=3, output_dim_cls=3):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.fc_reg = nn.Linear(hidden_dim, output_dim_reg)  # Regression output: RULs
        self.fc_cls = nn.Linear(hidden_dim, output_dim_cls)  # Classification output: fault probs

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last time step
        out = self.dropout(out)

        rul_out = self.fc_reg(out)
        fault_out = torch.sigmoid(self.fc_cls(out))  # Binary classification

        return rul_out, fault_out
