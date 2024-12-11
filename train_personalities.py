import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Step 1: Load and Preprocess the Data
# Assuming the data is loaded into a DataFrame called df
df = pd.read_csv('../Data/data-final.csv', sep='\t', quoting=3)

# Clean the dataset by removing extra quotation marks
df.columns = df.columns.str.replace('"', '').str.strip()
df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)

# Convert personality scores to numeric
df.iloc[:, :50] = df.iloc[:, :50].apply(pd.to_numeric, errors='coerce')

# Reverse scoring for negatively keyed items
negatively_keyed = ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',
                    'EST2', 'EST4',
                    'AGR1', 'AGR3', 'AGR5', 'AGR7',
                    'CSN2', 'CSN4', 'CSN6', 'CSN8',
                    'OPN2', 'OPN4', 'OPN6']
for col in negatively_keyed:
    df[col] = 6 - df[col]

# Drop rows with missing values
df = df.dropna()

# Step 2: Aggregate trait scores
df['EXT'] = df[['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10']].mean(axis=1)
df['AGR'] = df[['AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10']].mean(axis=1)
df['EST'] = df[['EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10']].mean(axis=1)
df['CSN'] = df[['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10']].mean(axis=1)
df['OPN'] = df[['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10']].mean(axis=1)

# Use the aggregated traits as features
features = ['EXT', 'AGR', 'EST', 'CSN', 'OPN']
X = df[features].values

# Step 3: Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 4: Train the model for each personality trait
models = {}
trait_columns = ['EXT', 'AGR', 'EST', 'CSN', 'OPN']
scaler = StandardScaler()

# Create a directory to save model checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

for trait in trait_columns:
    y = df[trait].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Print train loss
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss for {trait}: {loss.item()}')
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{trait}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
    
    models[trait] = model

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f'Test Loss for {trait}: {test_loss.item()}')
