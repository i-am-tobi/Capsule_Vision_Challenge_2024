import torch
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from DataModule import DataModule
from Model import Model

# HyperParameters
num_numerical_features = 186
growth_rate = 24
num_layers = 16
reduction = 0.5
batch_size, epochs = 64, 200

# Transformations for the images
image_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize dataset
train_data = DataModule(csv_file="merged_CSV_training_file.csv", image_dir="training/", transform=image_transforms)
val_data = DataModule(csv_file="horizontally_stacked_features (1).csv", image_dir="validation/", transform=image_transforms)

# Initialize model
model = Model(num_numerical_features=num_numerical_features, num_classes=10, growth_rate=growth_rate,
              num_layers=num_layers, reduction=reduction).cuda()

# Function to train/validate the model
def train_val(net, data_loader, train_optimizer, epoch, epochs):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for image, mask, features, target in data_bar:
            image, mask, features, target = image.cuda(non_blocking=True), mask.cuda(non_blocking=True), features.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(image, mask, features)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += image.size(0)
            total_loss += loss.item() * image.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

def validate(net, data_loader):
    net.eval()  # Set the model to evaluation mode
    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)

    with torch.no_grad():  # Disable gradient calculation
        for image, mask, features, target in data_bar:
            image, mask, features, target = image.cuda(non_blocking=True), mask.cuda(non_blocking=True), features.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(image, mask, features)  # Forward pass
            loss = loss_criterion(out, target)

            total_num += image.size(0)
            total_loss += loss.item() * image.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('Validation Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format(total_loss / total_num, total_correct_1 / total_num * 100,
                                             total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

# Function for training and validation with saving the best model and logging
def train_and_validate(net, train_loader, val_loader, train_optimizer, epochs, save_path="best_model.pth", log_path="training_logs.csv"):
    best_val_acc = 0.0  # To track the best validation accuracy
    log_data = {'epoch': [], 'train_loss': [], 'train_acc@1': [], 'train_acc@5': [], 'val_loss': [], 'val_acc@1': [], 'val_acc@5': []}
    
    for epoch in range(epochs):
        # Training
        train_loss, train_acc_1, train_acc_5 = train_val(net, train_loader, train_optimizer, epoch + 1, epochs)
        
        # Validation
        val_loss, val_acc_1, val_acc_5 = validate(net, val_loader)
        
        # Print both training and validation accuracies for the current epoch
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train ACC@1: {train_acc_1:.2f}%, Train ACC@5: {train_acc_5:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val ACC@1: {val_acc_1:.2f}%, Val ACC@5: {val_acc_5:.2f}%")
        
        # Save the model if it has the best validation accuracy
        if val_acc_1 > best_val_acc:
            best_val_acc = val_acc_1
            torch.save(net.state_dict(), save_path)
            print(f"Best model saved with Val ACC@1: {val_acc_1:.2f}%\n")
        
        # Log data for this epoch
        log_data['epoch'].append(epoch + 1)
        log_data['train_loss'].append(train_loss)
        log_data['train_acc@1'].append(train_acc_1)
        log_data['train_acc@5'].append(train_acc_5)
        log_data['val_loss'].append(val_loss)
        log_data['val_acc@1'].append(val_acc_1)
        log_data['val_acc@5'].append(val_acc_5)

        # Save logs after every epoch
        logs_df = pd.DataFrame(log_data)
        logs_df.to_csv(log_path, index=False)

    print(f"Training completed. Best Val ACC@1: {best_val_acc:.2f}%")

# Weighted sampler to handle imbalanced classes
#train_sampler = get_weighted_sampler(train_data.dataframe)
#val_sampler = get_weighted_sampler(val_data.dataframe)

# Initialize data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data,batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
loss_criterion = nn.CrossEntropyLoss()

train_and_validate(model, train_loader, val_loader, optimizer, epochs=epochs)

