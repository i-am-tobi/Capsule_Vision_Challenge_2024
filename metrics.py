import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             f1_score, recall_score, average_precision_score,
                             confusion_matrix, classification_report, roc_curve, auc)
import matplotlib.pyplot as plt

def evaluate_on_validation(net, data_loader):
    net.eval()  # Set the model to evaluation mode
    class_labels = ['Lymphangiectasia', 'Polyp', 'Angioectasia', 'Bleeding', 'Erosion',
                    'Erythema', 'Foreign Body', 'Ulcer', 'Worm', 'Normal']

    # Containers to store predictions and labels
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():  # Disable gradient calculation for faster performance
        for image, mask, features, label in tqdm(data_loader):
            image, mask, features = image.cuda(), mask.cuda(), features.cuda()

            # Model inference
            output = net(image, mask, features)
            predictions = torch.softmax(output, dim=1)  # Get probabilities

            # Get predicted class index and actual labels for the entire batch
            batch_pred_indices = output.argmax(dim=-1).cpu().numpy()
            batch_labels = label.cpu().numpy()
            batch_probs = predictions.cpu().numpy()

            # Store predictions and labels for metric calculation
            all_labels.extend(batch_labels)
            all_predictions.extend(batch_pred_indices)
            all_probs.extend(batch_probs)

    # Convert to numpy arrays for sklearn metric calculations
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Calculate per-class metrics and plot ROC curve for each class
    metrics = {}
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_labels):
        y_true = (all_labels == i).astype(int)  # Binary ground truth for this class
        y_prob = all_probs[:, i]  # Probability of being in this class

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

        # Calculate other metrics
        balanced_acc = balanced_accuracy_score(y_true, (y_prob > 0.5).astype(int))
        f1 = f1_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)
        sensitivity = recall_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)
        average_precision = average_precision_score(y_true, y_prob) if np.sum(y_true) > 0 else np.nan

        # Store metrics
        metrics[class_name] = {
            "Balanced Accuracy": balanced_acc,
            "AUC-ROC": roc_auc,
            "F1 Score": f1,
            "Sensitivity": sensitivity,
            "Average Precision": average_precision
        }

    # Plot configuration
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Each Class")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Print metrics
    print("\nPer-Class Metrics:")
    for class_name, m in metrics.items():
        print(f"{class_name}: {m}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_labels))
