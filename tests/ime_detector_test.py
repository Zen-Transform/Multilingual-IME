import torch
import matplotlib.pyplot as plt
from datetime import datetime

from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix

from multilingual_ime.ime_detector import IMEDetectorOneHot

if __name__ == "__main__":
    DATA_AND_LABEL_SPLITTER = "\t"
    NUM_OF_CLASSES = 2
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")
    TEST_FILE_PATH = "Datasets\\Test_Datasets\\labeled_bopomofo_0_test.txt"
    TEST_RESULT_FILE_PATH = f"reports\\ime_detector_test_result_{TIMESTAMP}.txt"

    ime_detector = IMEDetectorOneHot('multilingual_ime\\src\\model_dump\\one_hot_dl_model_bopomofo_2024-07-14.pkl', device="cpu")
    accuracy = MulticlassAccuracy(num_classes=NUM_OF_CLASSES)
    precision = MulticlassF1Score(num_classes=NUM_OF_CLASSES)
    recall = MulticlassPrecision(num_classes=NUM_OF_CLASSES)
    f1_score = MulticlassRecall(num_classes=NUM_OF_CLASSES)
    confusion_matrix = MulticlassConfusionMatrix(num_classes=NUM_OF_CLASSES)

    with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
        test_data = [line.strip().split(DATA_AND_LABEL_SPLITTER) for line in f.readlines()][:100]
        test_data = [(line[0], line[1]) for line in test_data if len(line) == 2]

    with tqdm(total=len(test_data), desc="Testing IME Detector") as pbar:
        for test_case, label in test_data:
            result = torch.tensor([int(ime_detector.predict(test_case))]).clone().detach()
            label = torch.tensor([int(label)]).clone().detach()
            print("result", result)
            print("label", label)

            accuracy.update(result, label)
            precision.update(result, label)
            recall.update(result, label)
            f1_score.update(result, label)
            confusion_matrix.update(result, label)
            pbar.update()

    print("IME Detector Test Result:")
    print(f"Total Test Cases: {len(test_data)}")
    print(f"Accuracy: {accuracy.compute()}")
    print(f"Precision: {precision.compute()}")
    print(f"Recall: {recall.compute()}")
    print(f"F1: {f1_score.compute()}")
    print(f"Confusion Matrix: {confusion_matrix.compute()}")
    fig, ax = confusion_matrix.plot()
    ax.set_title("Train Confusion Matrix")
    fig.show()
    plt.show(block=True)
    plt.savefig("reports\\ime_detector_test_confusion_matrix_{}.png".format(TIMESTAMP))

    with open(TEST_RESULT_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("IME Detector Test Result:\n" + \
                f"Total Test Cases: {len(test_data)}\n" + \
                f"Accuracy: {accuracy.compute()}\n" + \
                f"Precision: {precision.compute()}\n" + \
                f"Recall: {recall.compute()}\n" + \
                f"F1: {f1_score.compute()}\n" + \
                f"Confusion Matrix: {confusion_matrix.compute()}\n"
        )
