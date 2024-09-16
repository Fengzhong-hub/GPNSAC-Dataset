import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import data_process
from Fast_KAN_model import FastKAN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Using the 'Agg' backend
import pickle


# 1. Defining a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 2. Loading Data
def load_data(file_path):

    label_all, min_values, max_values, label_encoders, dataNormalization = data_process.data_pre_process(file_path)
    for i in range(len(label_all)):
        if label_all[i] > 1:
            label_all[i] = label_all[i] - 1
    return dataNormalization, label_all, min_values, max_values, label_encoders

# 3. Divide the dataset
def split_data(data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


# Calculate accuracy
def calculate_accuracy(outputs, targets):
    # The dimension of outputs is (batch_size, num_classes), and torch.max is used to get the index of the maximum value as the predicted category
    _, preds = torch.max(outputs, dim=1)  # Find the class with the highest score in the output
    correct = (preds == targets).float().sum()  # Calculate the number of correct predictions
    accuracy = correct / targets.numel() * 100  # Calculate accuracy
    return accuracy




# 5. Define the training function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training the model
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, targets).item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        # Validate the model
        model.eval()
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print the results of each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {avg_train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")

    # Return loss and accuracy records
    return train_losses, train_accuracies, val_losses, val_accuracies

# 6. Validation Function
def validate(model, dataloader, criterion):
    running_loss, running_acc = 0.0, 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, targets).item()

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    return avg_loss, avg_acc

# 7. Test Function
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc

# 8. Plotting loss and accuracy curves
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs):
    plt.figure(figsize=(12, 5))

    # Plotting the loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Draw the accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig('training_validation_metrics.png')  # Save the image
    print("Image saved as 'training_validation_metrics.png'")



if __name__ == '__main__':
    # Loading the dataset
    file_path = '/home/wanghangyu/pythonProjects/testProject/data/output_2.log'
    data, labels, min_values, max_values, label_encoders = load_data(file_path)  # Dataset path
    # Save normalization parameters
    np.save('min_values_2.npy', min_values)
    np.save('max_values_2.npy', max_values)
    with open('label_encoders_2.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    # Divide the training set and test set
    X_train, X_test, y_train, y_test = split_data(data, labels, test_size=0.2)

    # Build DataLoader for training and testing sets
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Defining the model structure
    input_dim = X_train.shape[1]  # Input data dimensions
    output_dim = 4  # Output data dimensions
    hidden_layers = [input_dim, 64, 32, output_dim]  # Hidden layer structure

    # Instantiate Model
    model = FastKAN(hidden_layers)

    # Defining loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss() # Fork Entropy Loss Function
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Using the Adam optimizer

    # Training and Validation
    epochs = 10
    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, epochs)

    # Save the model
    model_path = 'kanModel_2'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot loss and accuracy curves for training and validation sets
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs)

    # Load the model for testing
    loaded_model = FastKAN(hidden_layers)
    loaded_model.load_state_dict(torch.load(model_path))
    print("Load the saved model and start testing...")
    test_loss, test_acc = test_model(loaded_model, val_loader, criterion)
