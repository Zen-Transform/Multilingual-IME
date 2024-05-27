from multilingual_ime.ime_detector import IMEDetectorSVM
from SVMClassification import (
    custom_tokenizer_bopomofo,
    custom_tokenizer_cangjie,
    custom_tokenizer_pinyin,
)
from collections import Counter
import re
from tqdm import tqdm
import random
import multiprocessing

class IMESeparator:
    def __init__(self):
        # self.separator = separator
        self.my_bopomofo_detector = IMEDetectorSVM(
            "..\\Model_dump\\bopomofo_8adj.pkl",
            "..\\Model_dump\\vectorizer_bopomofo_8adj.pkl",
        )
        self.my_eng_detector = IMEDetectorSVM(
            "..\\Model_dump\\english_8adj.pkl",
            "..\\Model_dump\\vectorizer_english_8adj.pkl",
        )
        self.my_cangjie_detector = IMEDetectorSVM(
            "..\\Model_dump\\cangjie_8adj.pkl",
            "..\\Model_dump\\vectorizer_cangjie_8adj.pkl",
        )
        self.my_pinyin_detector = IMEDetectorSVM(
            "..\\Model_dump\\pinyin_8adj.pkl",
            "..\\Model_dump\\vectorizer_pinyin_8adj.pkl",
        )

    def separate(self, input_keystrokes: str) -> list[list[str, str]]:
        results = []
        for seclen in range(3, len(input_keystrokes) + 1):
            former = input_keystrokes[:seclen]
            latter = input_keystrokes[seclen:]
            for method, detector in [
                ("bopomofo", self.my_bopomofo_detector),
                ("english", self.my_eng_detector),
                ("cangjie", self.my_cangjie_detector),
                ("pinyin", self.my_pinyin_detector),
            ]:
                if method == "english":
                    if (
                    detector.predict_eng(former) == True
                    and detector.predict_eng(latter) == False
                    ):
                        former_method = method
                        for method, detector in [
                            ("bopomofo", self.my_bopomofo_detector),
                            ("english", self.my_eng_detector),
                            ("cangjie", self.my_cangjie_detector),
                            ("pinyin", self.my_pinyin_detector),
                        ]:
                            if method == "english":
                                if detector.predict_eng(latter) == True:
                                    results.append([(former_method, former), (method, latter)])
                            else:
                                if detector.predict(latter) == True:
                                    results.append([(former_method, former), (method, latter)])
                else:
                    if (
                        detector.predict(former) == True
                        and detector.predict(latter) == False
                    ):
                        former_method = method
                        for method, detector in [
                            ("bopomofo", self.my_bopomofo_detector),
                            ("english", self.my_eng_detector),
                            ("cangjie", self.my_cangjie_detector),
                            ("pinyin", self.my_pinyin_detector),
                        ]:
                            if method == "english":
                                if detector.predict_eng(latter) == True:
                                    results.append([(former_method, former), (method, latter)])
                            else:
                                if detector.predict(latter) == True:
                                    results.append([(former_method, former), (method, latter)])
        if len(results) == 0:
            results.append(
                [
                    ("bopomofo", input_keystrokes),
                    ("english", input_keystrokes),
                    ("cangjie", input_keystrokes),
                    ("pinyin", input_keystrokes),
                ]
            )
        return results


def read_labels(file_path):
    labels = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                input_text, label_str = line.split("\t©©©\t")
                labels.append([input_text, label_str.strip()])
            except ValueError:
                pass
    return labels

def worker(label_chunk, my_separator, results_queue):
    match_count = 0
    temp_count = 1
    temp_accuracy = 0
    with open("all_old_Separator_prediction_summary.txt", "a", encoding="utf-8") as f:
        for label in tqdm(label_chunk, desc="Processing", unit=" line"):
            input_text = label[0]
            expected_label_pairs = re.findall(r"\([^)]*\)", label[1])
            results = my_separator.separate(input_text)
            for result in results:
                temp_len = 0
                for pairs in expected_label_pairs:
                    if eval(pairs) in result:
                        temp_len += 1
                if temp_len == len(expected_label_pairs):
                    match_count += 1
                    f.write(str(temp_count) + ". Correct\n")
                    f.write("Input: %s\n" % input_text)
                    f.write("Expected: %s\n" % expected_label_pairs)
                    f.write("Result: %s\n" % results)
                    break
            if temp_len != len(expected_label_pairs):
                f.write(str(temp_count) + ". Incorrect\n")
                f.write("Input: %s\n" % input_text)
                f.write("Expected: %s\n" % expected_label_pairs)
                f.write("Result: %s\n" % results)
            temp_count += 1
    temp_accuracy = match_count / len(label_chunk)
    results_queue.put(temp_accuracy)

def check_accuracy(label_file, my_separator):
    labels = read_labels(label_file)
    labels = random.sample(labels, 5000)
    total_lines = len(labels)
    print("Total lines: %d" % total_lines)
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(labels) // num_processes
    chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]
    results_queue = multiprocessing.Queue()
    processes = []

    for chunk in chunks:
        p = multiprocessing.Process(target=worker, args=(chunk, my_separator, results_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    accuracies = []
    while not results_queue.empty():
        accuracies.append(results_queue.get())

    accuracy = sum(accuracies) / len(accuracies)
    return accuracy


# def check_accuracy(label_file, my_separator):
#     labels = read_labels(label_file)
#     labels = random.sample(labels, 5000)
#     total_lines = len(labels)
#     print("Total lines: %d" % total_lines)
#     match_count = 0
#     temp_count = 1
#     with open("all_old_Separator_prediction_summary.txt", "w", encoding="utf-8") as f:
#         for label in tqdm(labels, desc="Processing", unit=" line"):
#             input_text = label[0]
#             expected_label_pairs = re.findall(r"\([^)]*\)", label[1])
#             results = my_separator.separate(input_text)
#             for result in results:
#                 temp_len = 0
#                 for pairs in expected_label_pairs:
#                     if eval(pairs) in result:
#                         temp_len += 1
#                 if temp_len == len(expected_label_pairs):
#                     match_count += 1
#                     # print(results)
#                     # print("Check1: ", result, "Check2: ", expected_label_pairs)
#                     f.write(str(temp_count) + ". Correct\n")
#                     f.write("Input: %s\n" % input_text)
#                     f.write("Expected: %s\n" % expected_label_pairs)
#                     f.write("Result: %s\n" % results)
#                     break
#             if temp_len != len(expected_label_pairs):
#                 # print(results)
#                 # print("Check1: ", result, "Check2: ", expected_label_pairs)
#                 f.write(str(temp_count) + ". Incorrect\n")
#                 f.write("Input: %s\n" % input_text)
#                 f.write("Expected: %s\n" % expected_label_pairs)
#                 f.write("Result: %s\n" % results)
#             temp_count += 1
#         # print("Match count: %d" % match_count)
#     accuracy = match_count / total_lines
#     return accuracy


if __name__ == "__main__":
    my_separator = IMESeparator()
    accuracy = check_accuracy("..\\System_Test\\labled_mix_ime_r0-1.txt", my_separator)
    print("Accuracy: ", accuracy)
