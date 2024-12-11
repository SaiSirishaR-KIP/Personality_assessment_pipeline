import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

# Define a simple neural network model
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

# Load the model checkpoints
checkpoint_dir = './checkpoints'
models = {}
trait_columns = ['EXT', 'AGR', 'EST', 'CSN', 'OPN']
input_dim = 5  # Number of features

for trait in trait_columns:
    model = SimpleNN(input_dim=input_dim)
    checkpoint_path = os.path.join(checkpoint_dir, f'{trait}_epoch_100.pth')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    models[trait] = model

# Mapping between questionnaire text and trait labels
trait_mapping = {
    "I am the life of the party": "EXT1",
    "I don't talk a lot": "EXT2",
    "I feel comfortable around people": "EXT3",
    "I keep in the background": "EXT4",
    "I start conversations": "EXT5",
    "I have little to say": "EXT6",
    "I talk to a lot of different people at parties": "EXT7",
    "I don't like to draw attention to myself": "EXT8",
    "I don't mind being the center of attention.": "EXT9",
    "I am quiet around strangers": "EXT10",
    "I feel little concern for others": "AGR1",
    "I am interested in people": "AGR2",
    "I insult people": "AGR3",
    "I sympathize with others' feelings": "AGR4",
    "I am not interested in other people's problems": "AGR5",
    "I have a soft heart": "AGR6",
    "I am not really interested in others": "AGR7",
    "I take time out for others": "AGR8",
    "I feel others' emotions": "AGR9",
    "I make people feel at ease": "AGR10",
    "I am always prepared": "CSN1",
    "I leave my belongings around": "CSN2",
    "I pay attention to details": "CSN3",
    "I make a mess of things.": "CSN4",
    "I get chores done right away": "CSN5",
    "I often forget to put things back in their proper place": "CSN6",
    "I like order": "CSN7",
    "I shirk my duties": "CSN8",
    "I follow a schedule": "CSN9",
    "I am exacting in my work": "CSN10",
    "I get stressed out easily": "EST1",
    "I am relaxed most of the time": "EST2",
    "I worry about things": "EST3",
    "I seldom feel blue": "EST4",
    "I am easily disturbed": "EST5",
    "I get upset easily": "EST6",
    "I change my mood a lot": "EST7",
    "I have frequent mood swings": "EST8",
    "I get irritated easily": "EST9",
    "I often feel blue": "EST10",
    "I have a rich vocabulary": "OPN1",
    "I have difficulty understanding abstract ideas": "OPN2",
    "I have a vivid imagination": "OPN3",
    "I am not interested in abstract ideas": "OPN4",
    "I have excellent ideas": "OPN5",
    "I do not have a good imagination": "OPN6",
    "I am quick to understand things": "OPN7",
    "I use difficult words": "OPN8",
    "I spend time reflecting on things": "OPN9",
    "I am full of ideas": "OPN10"
}

def predict_personality(file_path):
    # Read the input Excel file
    new_data = pd.read_excel(file_path)

    # Map questionnaire text to trait labels
    mapped_columns = {column: trait_mapping[column] for column in new_data.columns if column in trait_mapping}
    new_data = new_data.rename(columns=mapped_columns)

    # Preprocess the new data
    new_data = new_data.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
    new_data.iloc[:, :50] = new_data.iloc[:, :50].apply(pd.to_numeric, errors='coerce')

    # Reverse scoring for negatively keyed items
    negatively_keyed = ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',
                        'EST2', 'EST4',
                        'AGR1', 'AGR3', 'AGR5', 'AGR7',
                        'CSN2', 'CSN4', 'CSN6', 'CSN8',
                        'OPN2', 'OPN4', 'OPN6']
    for col in negatively_keyed:
        if col in new_data.columns:
            new_data[col] = 6 - new_data[col]

    # Drop rows with missing values
    new_data = new_data.dropna()

    # Aggregate trait scores
    new_data['EXT'] = new_data[[f'EXT{i}' for i in range(1, 11) if f'EXT{i}' in new_data.columns]].mean(axis=1)
    new_data['AGR'] = new_data[[f'AGR{i}' for i in range(1, 11) if f'AGR{i}' in new_data.columns]].mean(axis=1)
    new_data['EST'] = new_data[[f'EST{i}' for i in range(1, 11) if f'EST{i}' in new_data.columns]].mean(axis=1)
    new_data['CSN'] = new_data[[f'CSN{i}' for i in range(1, 11) if f'CSN{i}' in new_data.columns]].mean(axis=1)
    new_data['OPN'] = new_data[[f'OPN{i}' for i in range(1, 11) if f'OPN{i}' in new_data.columns]].mean(axis=1)

    # Use the aggregated traits as features
    features = ['EXT', 'AGR', 'EST', 'CSN', 'OPN']
    X_new = new_data[features].values

    # Scale the features
    scaler = StandardScaler()
    X_new = scaler.fit_transform(X_new)

    # Convert to PyTorch tensor
    X_new = torch.tensor(X_new, dtype=torch.float32)

    # Predict the personality traits for each user
    predictions = {}
    for trait, model in models.items():
        with torch.no_grad():
            predictions[trait] = model(X_new).numpy().flatten()

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Print the predicted traits
    print("Predicted Traits:")
    print(predictions_df)

    # Find and print the top 2 dominant traits for each user
    top_2_traits = predictions_df.apply(lambda row: row.nlargest(2).index.tolist(), axis=1)
    print("\nTop 2 Dominant Traits:")
    print(top_2_traits)

# Example usage
predict_personality('../Data/Siri.xlsx')
