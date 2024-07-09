import os
import random

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from data_preprocess.keystroke_tokenizer import KeystrokeTokenizer

class KeystrokeDataset(Dataset):
    def __init__(self, data: list[torch.Tensor, torch.Tensor]):
        one_hot_keystokes = [d[0] for d in data]
        one_hot_targets = [d[1] for d in data]
        self.one_hot_keystokes = one_hot_keystokes
        self.one_hot_targets = one_hot_targets

    def __len__(self):
        return len(self.one_hot_keystokes)

    def __getitem__(self, idx):
        return self.one_hot_keystokes[idx], self.one_hot_targets[idx]


random.seed(42)
torch.manual_seed(42)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")


# DATA Configuration
NUM_OF_TRAIN_DATA = 100  # 6B data
NONE_ERROR_VS_ERROR_RATIO = 0.75
TRAIN_VAL_SPLIT_RATIO = 0.8
MAX_TOKEN_SIZE = 30

# Model Configuration
INPUT_SHAPE = MAX_TOKEN_SIZE * KeystrokeTokenizer.key_labels_length()
NUM_CLASSES = 2

# Training Configuration
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001


if __name__ == "__main__":
    Train_data_no_error_path = ".\\Datasets\\Train_Datasets\\labeled_bopomofo_0_train.txt"
    Train_data_with_error_path = ".\\Datasets\\Train_Datasets\\labeled_bopomofo_r0-1_train.txt" 

    if not os.path.exists(Train_data_no_error_path):
        raise FileNotFoundError(f"Train data not found at {Train_data_no_error_path}")
    if not os.path.exists(Train_data_with_error_path):
        raise FileNotFoundError(f"Train data not found at {Train_data_with_error_path}")

    with open(Train_data_no_error_path, "r", encoding="utf-8") as f:
        Train_data_no_error = f.readlines()
    with open(Train_data_with_error_path, "r", encoding="utf-8") as f:
        Train_data_with_error = f.readlines()

    training_datas = random.sample(Train_data_no_error,   int(NUM_OF_TRAIN_DATA * NONE_ERROR_VS_ERROR_RATIO)) \
                   + random.sample(Train_data_with_error, int(NUM_OF_TRAIN_DATA * (1 - NONE_ERROR_VS_ERROR_RATIO)))
    
    # format data to one-hot encoding and Tensor
    train_data_tensor = []
    for train_example in training_datas:
        keystoke, target = train_example.split("\t")

        token_ids = KeystrokeTokenizer.token_to_ids(KeystrokeTokenizer.tokenize(keystoke))
        token_ids = token_ids[:MAX_TOKEN_SIZE]  # truncate to MAX_TOKEN_SIZE 
        token_ids += [0] * (MAX_TOKEN_SIZE - len(token_ids))  # padding


        one_hot_keystrokes = torch.zeros(MAX_TOKEN_SIZE, KeystrokeTokenizer.key_labels_length()) \
                           + torch.eye(KeystrokeTokenizer.key_labels_length())[token_ids]
        one_hot_keystrokes = one_hot_keystrokes.view(-1)  # flatten
        one_hot_targets = torch.tensor([1, 0], dtype=torch.float32) if target == "0" else torch.tensor([0, 1], dtype=torch.float32)
        
        assert INPUT_SHAPE == list(one_hot_keystrokes.view(-1).shape)[0], f"{INPUT_SHAPE} != {list(one_hot_keystrokes.view(-1).shape)[0]}"
        train_data_tensor.append([one_hot_keystrokes, one_hot_targets])

    print("Data loaded")
    train_data, val_data = torch.utils.data.random_split(train_data_tensor, [TRAIN_VAL_SPLIT_RATIO, 1 - TRAIN_VAL_SPLIT_RATIO])

    train_data = KeystrokeDataset(train_data)
    val_data = KeystrokeDataset(val_data)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = nn.Sequential(
            nn.Linear(INPUT_SHAPE, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
            nn.Softmax(dim=1),
        )
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()                                        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Start training")
    with tqdm(range(EPOCHS)) as pbar:
        for epoch in range(EPOCHS):
            pbar.desc = f"Epoch {epoch+1}/{EPOCHS}"

            # Training
            model.train()
            train_correct = 0
            train_total = 0
            train_total_loss = 0
            for batch_inputs, batch_labels in train_data_loader:
                
                batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 0)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

            pbar.set_postfix(training_loss=train_total_loss, train_acc=train_correct / train_total)


            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_total_loss = 0
            with torch.no_grad():
                for batch_inputs, batch_labels in val_data_loader:
                    batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_labels)
                    val_total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 0)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            pbar.set_postfix(val_loss=val_total_loss, val_acc=val_correct / val_total)
            pbar.update(1)
