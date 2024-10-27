import os
import random
from datetime import datetime
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy

from .keystroke_tokenizer import KeystrokeTokenizer
from .deep_models import LanguageDetectorModel, TokenDetectorModel


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
NUM_OF_TRAIN_DATA = 600000  # 6K data = 600,000
NONE_ERROR_VS_ERROR_RATIO = 0.75
TRAIN_VAL_SPLIT_RATIO = 0.8
MAX_TOKEN_SIZE = 30

# Model Configuration
MODEL_PREFIX = "one_hot_dl_token_model"
LANGUAGE = "pinyin"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
INPUT_SHAPE = MAX_TOKEN_SIZE * KeystrokeTokenizer.key_labels_length()
NUM_CLASSES = 2
MODEL_SAVE_PATH = f".\\models\\{MODEL_PREFIX}_{LANGUAGE}_{TIMESTAMP}.pth"

# Training Configuration
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
SKIP_VALIDATION = False


if __name__ == "__main__":
    Train_data_no_error_path = (
        f".\\Datasets\\Train_Datasets\\labeled_wlen1_{LANGUAGE}_0_train.txt"
    )
    Train_data_with_error_path = (
        f".\\Datasets\\Train_Datasets\\labeled_wlen1_{LANGUAGE}_r0-1_train.txt"
    )

    if not os.path.exists(Train_data_no_error_path):
        raise FileNotFoundError(f"Train data not found at {Train_data_no_error_path}")
    if not os.path.exists(Train_data_with_error_path):
        raise FileNotFoundError(f"Train data not found at {Train_data_with_error_path}")

    with open(Train_data_no_error_path, "r", encoding="utf-8") as f:
        Train_data_no_error = f.readlines()
    with open(Train_data_with_error_path, "r", encoding="utf-8") as f:
        Train_data_with_error = f.readlines()

    training_datas = random.sample(
        Train_data_no_error, int(NUM_OF_TRAIN_DATA * NONE_ERROR_VS_ERROR_RATIO)
    ) + random.sample(
        Train_data_with_error, int(NUM_OF_TRAIN_DATA * (1 - NONE_ERROR_VS_ERROR_RATIO))
    )

    # format data to one-hot encoding and Tensor
    train_data_tensor = []
    with tqdm(training_datas) as pbar:
        for train_example in training_datas:
            keystoke, target = train_example.strip("\n").split("\t")
            token_ids = KeystrokeTokenizer.token_to_ids(
                KeystrokeTokenizer.tokenize(keystoke)
            )
            token_ids = token_ids[:MAX_TOKEN_SIZE]  # truncate to MAX_TOKEN_SIZE
            token_ids += [0] * (MAX_TOKEN_SIZE - len(token_ids))  # padding

            one_hot_keystrokes = (
                torch.zeros(MAX_TOKEN_SIZE, KeystrokeTokenizer.key_labels_length())
                + torch.eye(KeystrokeTokenizer.key_labels_length())[token_ids]
            )
            one_hot_keystrokes = one_hot_keystrokes.view(-1)  # flatten
            one_hot_targets = (
                torch.tensor([0], dtype=torch.float32)
                if target == "0"
                else torch.tensor([1], dtype=torch.float32)
            )

            assert (
                INPUT_SHAPE == list(one_hot_keystrokes.view(-1).shape)[0]
            ), f"{INPUT_SHAPE} != {list(one_hot_keystrokes.view(-1).shape)[0]}"
            train_data_tensor.append([one_hot_keystrokes, one_hot_targets])
            pbar.update(1)

    print("Data loaded")
    train_data, val_data = torch.utils.data.random_split(
        train_data_tensor, [TRAIN_VAL_SPLIT_RATIO, 1 - TRAIN_VAL_SPLIT_RATIO]
    )

    train_data = KeystrokeDataset(train_data)
    val_data = KeystrokeDataset(val_data)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = TokenDetectorModel(input_shape=INPUT_SHAPE, num_classes=1)
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    fig = None

    print("Start training")
    best_val_acc = 0
    for epoch in range(EPOCHS):
        print(f"============ Epoch {epoch+1}/{EPOCHS} ============")

        # Training
        model.train()
        train_predicts = []
        train_labels = []
        train_loss = 0
        with tqdm(train_data_loader) as train_pbar:
            for i, (batch_X, batch_Y) in enumerate(train_data_loader):
                train_pbar.set_description(f"Train {i+1}/{len(train_data_loader)}")
                batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                loss.backward()
                optimizer.step()

                # predictions = torch.argmax(outputs.data, dim=-1)
                # labels = torch.argmax(batch_Y, dim=-1)
                predictions = torch.round(outputs)
                # print("prediction",predictions)
                # print("labels",batch_Y)
                labels = batch_Y
                batch_loss = loss.item()
                batch_acc = MulticlassAccuracy(num_classes=NUM_CLASSES).to(DEVICE)

                train_predicts.append(predictions)
                train_labels.append(labels)
                train_loss += batch_loss

                train_pbar.set_postfix(
                    batch_loss=batch_loss, batch_acc=batch_acc(predictions, labels)
                )
                train_pbar.update(1)

        train_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(DEVICE)
        print(
            f"Training loss: {train_loss}, Training accuracy: {train_metric(torch.cat(train_predicts), torch.cat(train_labels))}"
        )

        if SKIP_VALIDATION:
            continue

        # Validation
        model.eval()
        val_predicts = []
        val_labels = []
        val_loss = 0
        with tqdm(val_data_loader) as pbar:
            with torch.no_grad():
                for i, (batch_X, batch_Y) in enumerate(val_data_loader):
                    pbar.set_description(f"Val {i+1}/{len(val_data_loader)}")
                    batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_Y)

                    # predictions = torch.argmax(outputs.data, dim=-1)
                    # labels = torch.argmax(batch_Y, dim=-1)
                    predictions = torch.round(outputs)
                    labels = batch_Y
                    batch_val_loss = loss.item()
                    batch_val_acc = MulticlassAccuracy(num_classes=NUM_CLASSES).to(
                        DEVICE
                    )

                    val_predicts.append(predictions)
                    val_labels.append(labels)
                    val_loss += batch_val_loss

                    pbar.set_postfix(
                        batch_val_loss=batch_val_loss,
                        batch_val_acc=batch_val_acc(predictions, labels),
                    )
                    pbar.update(1)
        val_acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(DEVICE)
        val_acc = val_acc_metric(torch.cat(val_predicts), torch.cat(val_labels))
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Best model saved with accuracy {best_val_acc}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # joblib.dump(model, MODEL_SAVE_PATH)

        print("====================================\n")

    confusion_matrix = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(DEVICE)
    confusion_matrix(torch.cat(train_predicts), torch.cat(train_labels))
    print(confusion_matrix.compute())
    fig, ax = confusion_matrix.plot()
    ax.set_title("Train Confusion Matrix")
    fig.show()
    plt.show(block=True)
