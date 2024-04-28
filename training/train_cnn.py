import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# Step 1: Load the dataset
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        # Return the total number of windows
        return len(self.data_frame) - 32

    def __getitem__(self, idx):
        # Get the 32x8 input chunk and the 33rd sample as the target
        input_chunk = self.data_frame.iloc[idx:idx+32].values.astype('float32')
        target_sample = self.data_frame.iloc[idx+32].values.astype('float32')
        # Reshape input to [1, 32, 8] to treat as single-channel image
        input_chunk = input_chunk.reshape(1, 32, 8)
        return input_chunk, target_sample

# Replace 'your_dataset.csv' with the path to your actual CSV file
dataset = CSVDataset('your_dataset.csv')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Step 2: Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 2, 8)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        return x

model = SimpleCNN()

# Step 3: Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Step 4: Training Loop
num_epochs = 5  # You can adjust this
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # Print every 1000 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0

print('Finished Training')

