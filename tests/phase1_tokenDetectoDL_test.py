from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import random
import matplotlib.pyplot as plt
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix,
)

from multilingual_ime.ime import PINYIN_IME, BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME
from multilingual_ime.ime_detector import IMETokenDetectorDL


# Test Fix Parameters
BOPOMOFO_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent.parent
    / "multilingual_ime"
    / "src"
    / "models"
    / "one_hot_dl_token_model_bopomofo_2024-10-27.pth"
)
CANGJIE_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent.parent
    / "multilingual_ime"
    / "src"
    / "models"
    / "one_hot_dl_token_model_cangjie_2024-10-27.pth"
)
PINYIN_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent.parent
    / "multilingual_ime"
    / "src"
    / "models"
    / "one_hot_dl_token_model_pinyin_2024-10-27.pth"
)
ENGLISH_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent.parent
    / "multilingual_ime"
    / "src"
    / "models"
    / "one_hot_dl_token_model_english_2024-10-27.pth"
)

IME_DETECTORDL_PATH_MAP = {
    BOPOMOFO_IME: BOPOMOFO_IME_TOKEN_DETECTOR_MODEL_PATH,
    CANGJIE_IME: CANGJIE_IME_TOKEN_DETECTOR_MODEL_PATH,
    PINYIN_IME: PINYIN_IME_TOKEN_DETECTOR_MODEL_PATH,
    ENGLISH_IME: ENGLISH_IME_TOKEN_DETECTOR_MODEL_PATH,
}

DATA_AND_LABEL_SPLITTER = "\t"
NUM_OF_CLASSES = 2
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")


def error_rate_to_string(error_rate: float) -> str:
    if error_rate == 0:
        return "0"
    return "r" + str(error_rate).replace(".", "-")


if __name__ == "__main__":
    # Test Parameters
    TEST_CASES_NUMBER = 10000
    TEST_ERROR_RATE = 0
    TEST_IMES = [BOPOMOFO_IME, CANGJIE_IME, PINYIN_IME, ENGLISH_IME]
    # Settings
    SHOW_PLOT_AND_WAIT = True


    error_rate_str = error_rate_to_string(TEST_ERROR_RATE)

    for ime in TEST_IMES:
        accuracy_metric = BinaryAccuracy()
        precision_metric = BinaryF1Score()
        recall_metric = BinaryPrecision()
        f1_score_metric = BinaryRecall()
        confusion_matrix = BinaryConfusionMatrix()

        TEST_FILE_PATH = (
            f"Datasets\\Test_Datasets\\labeled_{ime}_{error_rate_str}_test.txt"
        )
        TEST_RESULT_IMAGE_PATH = f"reports\\tokenDetectorDL_{ime}_{error_rate_str}_test_result_{TIMESTAMP}.png"

        ime_detectorDL = IMETokenDetectorDL(
            IME_DETECTORDL_PATH_MAP[ime],
            device="cuda",
        )

        print("Loading Test Data")
        with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
            test_data = [
                line.strip().split(DATA_AND_LABEL_SPLITTER) for line in f.readlines()
            ]
            random.shuffle(test_data)
            test_data = [(line[0], line[1]) for line in test_data if len(line) == 2][
                :TEST_CASES_NUMBER
            ]

        with tqdm(total=len(test_data), desc=f"Testing IME DetectorDL {ime}") as pbar:
            all_labels = []
            all_preds = []

            for test_case, y_label in test_data:
                y_pred = torch.tensor([int(ime_detectorDL.predict(test_case))])
                y_label = torch.tensor([int(y_label)])

                all_labels.append(y_label)
                all_preds.append(y_pred)
                pbar.update()

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        print("all_labels", all_labels, all_labels.shape, all_labels.dtype)
        print("all_preds", all_preds, all_preds.shape, all_preds.dtype)

        accuracy = accuracy_metric(all_preds, all_labels)
        precision = precision_metric(all_preds, all_labels)
        recall = recall_metric(all_preds, all_labels)
        f1_score = f1_score_metric(all_preds, all_labels)
        confusion_matrix(all_preds, all_labels)

        # Draw Image
        fig, ax = confusion_matrix.plot()
        ax.set_title(f"IME DetectorDL: {ime} \n Confusion Matrix")

        text_x_pos = NUM_OF_CLASSES
        plt.text(
            text_x_pos,
            0.2,
            f"Total Test Cases: {len(test_data)}\n"
            + f"Error Rate: {TEST_ERROR_RATE}\n"
            + f"Accuracy: {accuracy}\n"
            + f"Precision: {precision}\n"
            + f"Recall: {recall}\n"
            + f"F1: {f1_score}\n",
        )
        plt.savefig(TEST_RESULT_IMAGE_PATH)
        plt.show(block=SHOW_PLOT_AND_WAIT)
