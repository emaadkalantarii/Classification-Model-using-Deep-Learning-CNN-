import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

MAX_SEQ_LENGTH = 150  

INPUT_FEATURES = 2
HIDDEN_SIZE = 128     # GRU hidden size per direction
NUM_RNN_LAYERS = 2    # Number of GRU layers
NUM_CLASSES = 4
DROPOUT_PROB = 0.25   # Dropout probability

CLASS_SUBDIRECTORIES_MAP = {'human': 0, 'gan': 1, 'sdt': 2, 'vae': 3} 

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
    

def preprocess_single_signature(csv_path, max_seq_len, input_features):
    try:
        df = pd.read_csv(csv_path, sep=' ', header=0, names=['X', 'Y'], engine='python')
        df['X'] = pd.to_numeric(df['X'], errors='coerce')
        df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
        df.dropna(inplace=True)
        processed_sequence = np.zeros((max_seq_len, input_features), dtype=np.float32)
        if not df.empty:
            current_coords = df[['X', 'Y']].values.astype(np.float32)
            if current_coords.shape[0] > 1:
                scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
                current_coords[:, 0] = scaler_x.fit_transform(current_coords[:, [0]]).flatten()
                current_coords[:, 1] = scaler_y.fit_transform(current_coords[:, [1]]).flatten()
            elif current_coords.shape[0] == 1:
                current_coords[:, :] = 0.0 
            len_to_copy = min(len(current_coords), max_seq_len)
            if len_to_copy > 0:
                processed_sequence[:len_to_copy, :] = current_coords[:len_to_copy, :]
    except Exception as e:
        print(f"Note: Error processing {os.path.basename(csv_path)} for eval: {e}. Using zero sequence.")
        processed_sequence = np.zeros((max_seq_len, input_features), dtype=np.float32)
    return torch.tensor(processed_sequence, dtype=torch.float32)


def load_and_predict(directory, model_file):
    """
    # The `directory` argument is a folder with the same structure as the provided dataset. Example:
    # /path/to/some/signatures
    #  |_human
    #     |_ 001g01.csv
    #     |_ 001g02.csv
    #     |_ ...
    #  |_gan
    #     |_ 001g01.csv
    #     |_ 001g02.csv
    #     |_ ...
    #  |_sdt
    #     |_ 001g01.csv
    #     |_ 001g02.csv
    #     |_ ...
    #  |_vae
    #     |_ 001g01.csv
    #     |_ 001g02.csv
    #     |_ ...
    #
    # The `model_file` argument is a trained model file, in `.pth` format.
    #
    # This function must implement the following steps:
    # 1. Read the data from the provided directory.
    # 2. Prepare the data according to whatever preprocessing pipeline was used during model training.
    # 3. Load the model checkpoint.
    # 4. Query the model with the data in order to get the predicted class probabilities.
    # 5. Convert probabilities to labels, like in the other assignment (but now there are 4 classes instead of 2).
    # 6. Return a dictionary where keys are absolute file paths and values are the predicted labels for each file. Example:
    # `{ '/path/to/some/signatures/human/001g01.csv': 0, '/path/to/some/signatures/sdt/001g02.csv': 2, ... }`.
   
    
    """
    # try to use GPU
    device = try_gpu()


    model = SignatureModel(INPUT_FEATURES, HIDDEN_SIZE, NUM_RNN_LAYERS, NUM_CLASSES, DROPOUT_PROB)
    try:
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
    except (TypeError, RuntimeError, AttributeError) as e: 
        print(f"Warn: Loading with 'weights_only=True' failed ('{e}'). Falling back.")
        state_dict = torch.load(model_file, map_location=device)
    except FileNotFoundError:
        print(f"Error: Model file '{model_file}' not found."); return {}
    except Exception as e:
         print(f"Error loading model checkpoint '{model_file}': {e}"); return {}
    
    try:
        model.load_state_dict(state_dict)
        model = model.to(device)
    except Exception as e:
        print(f"Error applying loaded state dict to model: {e}"); return {}

    model.eval() 

    all_csv_files_with_potential_duplicates = []
    for file_path in glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True):
        all_csv_files_with_potential_duplicates.append(os.path.abspath(file_path))
    all_csv_files = sorted(list(set(all_csv_files_with_potential_duplicates)))

    if not all_csv_files:
        print(f"Warn: No .csv files found in '{directory}' or its subdirectories."); return {}
        
    labels_dict = {} 
    eval_batch_size = 32 
    num_batches = (len(all_csv_files) + eval_batch_size - 1) // eval_batch_size

    with torch.no_grad(): 
        for i in range(num_batches):
            start_idx = i * eval_batch_size
            end_idx = min((i + 1) * eval_batch_size, len(all_csv_files))
            current_batch_file_paths = all_csv_files[start_idx:end_idx]

            batch_sequence_tensors = []
            valid_file_paths_in_batch = [] 

            for file_path in current_batch_file_paths:
                try:
                    sequence_tensor = preprocess_single_signature(file_path, MAX_SEQ_LENGTH, INPUT_FEATURES)
                    batch_sequence_tensors.append(sequence_tensor)
                    valid_file_paths_in_batch.append(file_path) 
                except Exception as e: 
                    print(f"Note: Skipping file {os.path.basename(file_path)} due to error: {e}")
                    continue 
            
            if not batch_sequence_tensors: continue 

            batch_input_tensor = torch.stack(batch_sequence_tensors).to(device)
            
            batch_output_logits = model(batch_input_tensor)
            
            _, predicted_labels_for_batch = torch.max(batch_output_logits, 1) 

            for j, file_path_key in enumerate(valid_file_paths_in_batch):
                labels_dict[file_path_key] = predicted_labels_for_batch[j].item()
    
    return labels_dict


if __name__ == "__main__":

    eval_directory = "./signatures" 
    model_filepath = "model.pth" 

    predictions = load_and_predict(eval_directory, model_filepath)
        
    if predictions:
        count_display = 0
        for path_disp, label_disp in predictions.items():
            try: display_path = os.path.relpath(path_disp)
            except ValueError: display_path = path_disp 
            print(f"  '{display_path}': {label_disp}")
            count_display += 1

            
            
            counter_correct = 0
            count = 0

            for file_path, predicted_label in predictions.items():
                try:
                    parent_dir_name = os.path.basename(os.path.dirname(file_path))
                    
                    if parent_dir_name in CLASS_SUBDIRECTORIES_MAP:
                        true_label = CLASS_SUBDIRECTORIES_MAP[parent_dir_name]
                        if predicted_label == true_label:
                            counter_correct += 1
                        count += 1
                    else:
                        pass 
                except Exception as e:
                    print(f"Error determining true label for {file_path}: {e}")

    print(f"\nTotal predictions: {len(predictions)}")
    print("Accuracy:  ", counter_correct * 1.0 / count)
    pass
