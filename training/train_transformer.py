import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import torch.nn.functional as F

# Ensure the directory for model saving exists
model_dir = 'models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print('Loading Training Data')
large_df = pd.read_csv('data/combined_dataset.csv')
print('Loaded Data.')
sessions = np.unique(large_df['session_id'])

# Iterate through sessions to construct a dataset.
for session in sessions[:1]:
    session_df = large_df[large_df['session_id'] == session]
    session_df = session_df.drop(['session_id', 'timestamp'], axis=1)

print('Creating Train / Validation data split.')
# Establish train-validation split
training_datapoints = []
training_labels = []
validation_datapoints = []
validation_labels = []

# Define the split point
split_point = int((len(session_df) - 33) * 0.8)

for i in range(len(session_df) - 33):
    datapoint = session_df[i:32+i].to_numpy()
    label = session_df.iloc[32+i+1].to_numpy()
    
    if i < split_point:
        training_datapoints.append(datapoint)
        training_labels.append(label)
    else:
        validation_datapoints.append(datapoint)
        validation_labels.append(label)

print('Train Validation Data Created.')

class ConvTransformerModel(nn.Module):
    def __init__(self):
        super(ConvTransformerModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        transformer_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.fc1 = nn.Linear(16 * 32, 8)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

# Convert lists of numpy arrays to PyTorch tensors
train_inputs = torch.tensor(training_datapoints, dtype=torch.float32)
train_labels = torch.tensor(training_labels, dtype=torch.float32).squeeze(1)

valid_inputs = torch.tensor(validation_datapoints, dtype=torch.float32)
valid_labels = torch.tensor(validation_labels, dtype=torch.float32).squeeze(1)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

valid_dataset = TensorDataset(valid_inputs, valid_labels)
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False)

# Define the model, optimizer, and loss function
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConvTransformerModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss().to(device)

print('Beginning Training!')
# Training and validation loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation phase
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            validation_loss += loss.item()

    print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}, Validation Loss: {validation_loss / len(valid_loader)}')

    # Save model after each epoch
    model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_path)

# Optionally, compute the overall MSE for the validation data after training
total_mse = 0
with torch.no_grad():
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        total_mse += criterion(output, target).item()

mean_mse = total_mse / len(valid_loader)
print(f'Mean Squared Error on Validation Set: {mean_mse}')
