from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import pandas as pd


class CellDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# d_model range：512 to 2048
# dropout range：0.1 to 0.5
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, num_layers, d_model=1024, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x)
        return x

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    average_loss = total_loss / total_count
    return average_loss

class EarlyStopping:
    def __init__(self, patience=5 , delta=1e-2, verbose=False, path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = 'best_model.pt'
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    return average_loss, accuracy

def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def predict_model(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())

    return predictions

def predict_and_save(model, data_loader, gene_expression_data, file_path, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    test_cell_names = gene_expression_data.index[data_loader.dataset.indices]

    predicted_clusters_df = pd.DataFrame({
        'Cell Name': test_cell_names,
        'Predicted Cluster': predictions
    })

    predicted_clusters_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

def predict_full_dataset(model, data, cell_names, file_path, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in DataLoader(data, batch_size=32):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    predicted_clusters_df = pd.DataFrame({
        'Cell Name': cell_names,
        'Predicted Cluster': predictions
    })
    predicted_clusters_df.to_csv(file_path, index=False)