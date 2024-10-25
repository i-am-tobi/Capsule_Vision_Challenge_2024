import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, num_class, dropout_prob=0.5):
        super(ClassificationHead, self).__init__()
        # Changed input features to 1024 to match the combined features
        self.fc1 = nn.Linear(1024, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)  # First dropout layer

        # Adding auxiliary layer
        self.fc_aux = nn.Linear(1024, 512)  # Auxiliary layer with 512 units
        self.relu_aux = nn.ReLU()
        self.dropout_aux = nn.Dropout(p=dropout_prob)  # Dropout after auxiliary layer

        self.fc2 = nn.Linear(512, num_class)  # Final layer for class predictions

    def forward(self, x):
        # First layer with dropout
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Auxiliary layer with dropout
        x = self.fc_aux(x)
        x = self.relu_aux(x)
        x = self.dropout_aux(x)

        # Final classification layer
        x = self.fc2(x)

        return x
