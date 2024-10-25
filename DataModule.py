import pandas as pd
from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import os
import numpy as np

class DataModule(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = self.dataframe.iloc[:, 93]  # Column with image file names

        # Combine all the numerical features
        self.numerical_data = pd.concat([self.dataframe.iloc[:, :93], self.dataframe.iloc[:, 94:-1]], axis=1).values

        # Standardize the numerical features
        self.numerical_mean = np.mean(self.numerical_data, axis=0)
        self.numerical_std = np.std(self.numerical_data, axis=0)
        self.numerical_std[self.numerical_std == 0] = 1  # Prevent division by zero

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths.iloc[idx])  # Load image
        image = Image.open(img_path)
        image_mask = torch.zeros(image.size[0], image.size[1])  # Create dummy mask

        labels = self.dataframe['label'].iloc[idx]  # Get the label

        if self.transform:
            image = self.transform(image)  # Apply transformations

        # Standardize the numerical features
        numerical_features = (self.numerical_data[idx] - self.numerical_mean) / self.numerical_std
        numerical_features = torch.tensor(numerical_features, dtype=torch.float32)

        label = torch.tensor(labels, dtype=torch.long)  # Convert label to tensor

        return image, image_mask, numerical_features, label

# Function to calculate weights and set up the WeightedRandomSampler
def get_weighted_sampler(dataframe):
    # Count the number of samples per class
    class_counts = dataframe['label'].value_counts().sort_index().values  # Class frequencies
    num_classes = len(class_counts)

    # Compute class weights (inverse of class frequency)
    class_weights = 1.0 / class_counts

    # Assign a weight to each sample based on its class
    sample_weights = [class_weights[label] for label in dataframe['label']]

    # Convert to tensor
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)

    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)

    return sampler

