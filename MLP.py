import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, embedding_dim=512, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)  # First dropout layer
        self.fc2 = nn.Linear(1024, embedding_dim)  # Embedding layer
        self.dropout2 = nn.Dropout(p=dropout_prob)  # Second dropout layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after first layer
        embedding = self.fc2(x)
        embedding = self.dropout2(embedding)  # Apply dropout after second layer
        return embedding
