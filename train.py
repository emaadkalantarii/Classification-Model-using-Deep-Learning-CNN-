import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

BASE_DATA_DIR = './signatures'
MODEL_SAVE_PATH = 'model.pth'
LABELS_MAP = {'human': 0, 'gan': 1, 'sdt': 2, 'vae': 3}
CLASS_NAMES = {v: k for k, v in LABELS_MAP.items()}

#Try to find the best Sequence length
#MAX_SEQ_LENGTH = 150 Loss: 0.3348, Accuracy: 80.21%
#MAX_SEQ_LENGTH = 570 Loss: 0.3520, Accuracy: 81.89%
#MAX_SEQ_LENGTH = 200 Loss: 0.4132, Accuracy: 78.74%
#MAX_SEQ_LENGTH = 400 Loss: 0.3812, Accuracy: 80.00%

MAX_SEQ_LENGTH = 150


INPUT_FEATURES = 2
HIDDEN_SIZE = 128     # GRU hidden size per direction
NUM_RNN_LAYERS = 2    # Number of GRU layers
NUM_CLASSES = 4
DROPOUT_PROB = 0.25   # Dropout probability

BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4   # L2 regularization
RANDOM_STATE = 42
GRADIENT_CLIP_VALUE = 1.0 # For gradient clipping

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
# as it is 70 for training, 15 for validation, so we have 15 for testing

class SignatureDataset(Dataset):
    def __init__(self, file_paths, labels, max_seq_len, input_features):
        self.file_paths = file_paths
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.input_features = input_features

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        sequence_data = self._load_and_preprocess_signature(file_path)
        return torch.tensor(sequence_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _load_and_preprocess_signature(self, csv_path):
        # Loads, normalizes (per-signature MinMax), pads/truncates
        try:
            df = pd.read_csv(csv_path, sep=' ', header=0, names=['X', 'Y'], engine='python')
            df['X'] = pd.to_numeric(df['X'], errors='coerce')
            df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
            df.dropna(inplace=True)
            processed_sequence = np.zeros((self.max_seq_len, self.input_features), dtype=np.float32)
            if not df.empty:
                current_coords = df[['X', 'Y']].values.astype(np.float32)
                if current_coords.shape[0] > 1:
                    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
                    current_coords[:, 0] = scaler_x.fit_transform(current_coords[:, [0]]).flatten()
                    current_coords[:, 1] = scaler_y.fit_transform(current_coords[:, [1]]).flatten()
                elif current_coords.shape[0] == 1: current_coords[:, :] = 0.0 
                len_to_copy = min(len(current_coords), self.max_seq_len)
                if len_to_copy > 0: processed_sequence[:len_to_copy, :] = current_coords[:len_to_copy, :]
        except Exception:
            processed_sequence = np.zeros((self.max_seq_len, self.input_features), dtype=np.float32)
        return processed_sequence


def load_data(data_dir, labels_map):
    file_paths, labels = [], []
    print(f"Loading data from: {data_dir}")
    for class_name, label_idx in labels_map.items():
        class_folder_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder_path):
            print(f"Warning: Dir not found for class '{class_name}': {class_folder_path}"); continue
        for file_name in os.listdir(class_folder_path):
            if file_name.endswith('.csv'):
                file_paths.append(os.path.join(class_folder_path, file_name)); labels.append(label_idx)
    if not file_paths: raise FileNotFoundError(f"No .csv files in subdirs of {data_dir}.")
    print(f"Total files found: {len(file_paths)}")
    for label_idx, class_name_val in CLASS_NAMES.items():
        print(f"  Class '{class_name_val}' (label {label_idx}): {labels.count(label_idx)} files")
    return file_paths, labels


class SignatureModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers, num_classes, dropout_prob):
        super(SignatureModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_rnn_layers, # num_rnn_layers = 2
                            batch_first=True, 
                            dropout=dropout_prob if num_rnn_layers > 1 else 0,
                            bidirectional=True)
        
        fc_input_size = hidden_size * 2 # Output from BiGRU
        self.dropout_head = nn.Dropout(dropout_prob) # Dropout before final layer
        self.fc_final = nn.Linear(fc_input_size, num_classes) # Direct classification layer

    def forward(self, x):
        rnn_out, h_n = self.rnn(x) 
        h_n_forward = h_n[-2, :, :] 
        h_n_backward = h_n[-1, :, :]
        out = torch.cat((h_n_forward, h_n_backward), dim=1)
        
        out = self.dropout_head(out)
        out = self.fc_final(out)
        return out


# try to use GPU if available
def try_gpu():
    if torch.cuda.is_available():
        print("Found GPU for training.")
        
        # try to use the GPU
        try:
            return torch.device('cuda')
        except:
            print("Error with GPU, falling back to CPU")
            return torch.device('cpu')
    else:
        print("No GPU found. Using CPU instead.")
        return torch.device('cpu')


def main():
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    device = try_gpu(); print(f"Using: {device}")
    
    file_paths, labels = load_data(BASE_DATA_DIR, LABELS_MAP)
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, train_size=TRAIN_RATIO, random_state=RANDOM_STATE, stratify=labels)
    val_test_ratio_from_total = 1.0 - TRAIN_RATIO
    relative_val_size_in_temp = VALIDATION_RATIO / val_test_ratio_from_total if val_test_ratio_from_total > 0 else 0.5
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, train_size=relative_val_size_in_temp, random_state=RANDOM_STATE, stratify=temp_labels)


    train_dataset = SignatureDataset(train_paths, train_labels, MAX_SEQ_LENGTH, INPUT_FEATURES)
    val_dataset = SignatureDataset(val_paths, val_labels, MAX_SEQ_LENGTH, INPUT_FEATURES)
    test_dataset = SignatureDataset(test_paths, test_labels, MAX_SEQ_LENGTH, INPUT_FEATURES)
    
    workers = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=(device.type=='cuda'))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=(device.type=='cuda'))
    
    model = SignatureModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_RNN_LAYERS, NUM_CLASSES, DROPOUT_PROB).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7) 

    best_val_loss = float('inf')
    epochs_no_improve, early_stopping_patience, lr_already_reduced_once = 0, 15, False

    for epoch in range(NUM_EPOCHS):
        model.train(); train_loss = 0.0
        for inputs, current_labels in train_loader:
            inputs, current_labels = inputs.to(device), current_labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, current_labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE); optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = train_loss / len(train_loader.dataset)
        
        model.eval(); val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, current_labels in val_loader:
                inputs, current_labels = inputs.to(device), current_labels.to(device)
                outputs = model(inputs); loss = criterion(outputs, current_labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += current_labels.size(0); correct += (predicted == current_labels).sum().item()
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total if total > 0 else 0
        
        old_lr = optimizer.param_groups[0]['lr']; scheduler.step(epoch_val_loss); new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            if not lr_already_reduced_once: print(f"  Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            lr_already_reduced_once = True

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {new_lr:.6f}')
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss; torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Saved new best model to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})"); epochs_no_improve = 0
        else: epochs_no_improve += 1

        if lr_already_reduced_once and epochs_no_improve >= early_stopping_patience:
             print(f"Early stopping: No Val Loss improvement for {epochs_no_improve} epochs after LR reduction."); break
             
    print("\nTraining finished.")

    try:
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    except (TypeError, RuntimeError) as e: 
        print(f"Warning: Loading 'weights_only=True' failed ('{e}'). Falling back."); state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
    except FileNotFoundError: print(f"Error: Model file {MODEL_SAVE_PATH} not found."); return
    model.load_state_dict(state_dict)

    model.eval(); test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, current_labels in test_loader:
            inputs, current_labels = inputs.to(device), current_labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, current_labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1); total += current_labels.size(0); correct += (predicted == current_labels).sum().item()
    final_test_loss = test_loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    final_test_acc = 100 * correct / total if total > 0 else 0
    print(f"Final Test Set Results - Loss: {final_test_loss:.4f}, Accuracy: {final_test_acc:.2f}%")

    if device.type == 'cuda': torch.cuda.empty_cache()

if __name__ == "__main__":
    main()