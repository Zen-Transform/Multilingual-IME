import os
import random
import json
from tqdm import tqdm
from datetime import datetime

from multilingual_ime.ime import IMEFactory
from data_preprocess.typo_generater import TypoGenerater
from data_preprocess.keystroke_converter import KeyStrokeConverter
from multilingual_ime.ime_converter import ChineseIMEConverter, EnglishIMEConverter
from multilingual_ime.ime import PINYIN_IME, BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME

random.seed(42)
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")
IMES = [BOPOMOFO_IME, CANGJIE_IME, PINYIN_IME, ENGLISH_IME]


TEST_CHINESE_FILE_PATH = f"Datasets\\Plain_Text_Datasets\\wlen1-1_cc100_test.txt"
TEST_ENGLISH_FILE_PATH = (
    f"Datasets\\Plain_Text_Datasets\\wlen1-1_English_multi_test.txt"
)

# Old Converter
test_bopomofo_converter = ChineseIMEConverter(
    "multilingual_ime\\src\\keystroke_mapping_dictionary\\bopomofo_dict_with_frequency.json"
)
test_cangjie_converter = ChineseIMEConverter(
    "multilingual_ime\\src\\keystroke_mapping_dictionary\\cangjie_dict_with_frequency.json"
)
test_pinyin_converter = ChineseIMEConverter(
    "multilingual_ime\\src\\keystroke_mapping_dictionary\\pinyin_dict_with_frequency.json"
)
test_english_converter = EnglishIMEConverter(
    "multilingual_ime\\src\\keystroke_mapping_dictionary\\english_dict_with_frequency.json"
)


def get_random_line_efficient(filename):
    # The Con of this function is that the longer the line is the higher chance it gets selected
    with open(filename, "rb") as file:
        file_size = os.path.getsize(filename)
        random_position = random.randint(0, file_size - 1)
        file.seek(random_position)

        # Move backwards to find the start of the line
        while file.tell() > 0 and file.read(1) != b"\n":
            file.seek(file.tell() - 2)

        # Read the line
        return file.readline().decode("utf-8").strip()


def load_tokens(file_path: str, max_size: int):
    lines = [get_random_line_efficient(file_path) for _ in range(max_size)]
    test_data = [line.strip().split("\t") for line in lines]
    test_data = [
        line[0]
        for line in test_data
        if len(line) == 2 and line[1] == "1" and len(line[0]) != 0
    ]
    random.shuffle(test_data)
    return test_data


def error_rate_to_string(error_rate: float) -> str:
    if error_rate == 0:
        return "0"
    return "r" + str(error_rate).replace(".", "-")


if __name__ == "__main__":
    # Test Parameters
    TEST_CASES_NUMBER = 10000
    TEST_ERROR_RATES = [0, 0.1, 0.05]
    TEST_MODE = "trie"  # "sqlite" or "trie"
    # Settings
    for TEST_ERROR_RATE in TEST_ERROR_RATES:

        error_rate_str = error_rate_to_string(TEST_ERROR_RATE)

        for ime_type in IMES:
            TEST_RESULT_JSON_PATH = f"reports\\phase2_converter_{TEST_MODE}_{ime_type}_{error_rate_str}_test_result_{TIMESTAMP}.json"
            if TEST_MODE == "sqlite":
                test_ime_handler = IMEFactory.create_ime(ime_type)
            elif TEST_MODE == "trie":
                if ime_type == BOPOMOFO_IME:
                    test_ime_converter = test_bopomofo_converter
                elif ime_type == CANGJIE_IME:
                    test_ime_converter = test_cangjie_converter
                elif ime_type == PINYIN_IME:
                    test_ime_converter = test_pinyin_converter
                elif ime_type == ENGLISH_IME:
                    test_ime_converter = test_english_converter
                else:
                    raise ValueError("Invalid IME Type")
            else:
                raise ValueError("Invalid Test Mode")

            print("Generating Test Data")
            test_data = []
            for _ in range(TEST_CASES_NUMBER):
                plain_text = get_random_line_efficient(
                    TEST_ENGLISH_FILE_PATH
                    if ime_type == ENGLISH_IME
                    else TEST_CHINESE_FILE_PATH
                )

                test_x = TypoGenerater.generate(
                    KeyStrokeConverter.convert(plain_text, convert_type=ime_type),
                    error_type="random",
                    error_rate=TEST_ERROR_RATE,
                )

                test_data.append((test_x, plain_text))

            # print(test_data)

            print("Start Testing")
            with tqdm(
                total=len(test_data),
                desc=f"Testing IME Converter {ime_type} with {TEST_MODE}",
            ) as pbar:
                correct = 0
                mrr_scores = []
                time_spends = []
                # logs = []

                for test_x, y_label in test_data:
                    if TEST_MODE == "sqlite":
                        start_time = datetime.now()
                        y_pred = test_ime_handler.get_token_candidates(test_x)
                        time_spend = datetime.now() - start_time
                        y_pred = [token[1] for token in y_pred]
                    else:
                        start_time = datetime.now()
                        y_pred = test_ime_converter.get_candidates(test_x)
                        time_spend = datetime.now() - start_time
                        y_pred = [token.word for token in y_pred]

                    if y_label in y_pred:
                        correct += 1
                        mrr_scores.append(1 / (y_pred.index(y_label) + 1))
                    else:
                        mrr_scores.append(0)

                    time_spends.append(time_spend.total_seconds())
                    # logs.append(
                    #     {
                    #         "Input": test_x,
                    #         "Output": y_pred,
                    #         "Correnct": y_label,
                    #         "MRR": mrr_scores[-1],
                    #         "Time Spend": time_spend.total_seconds(),
                    #     }
                    # )
                    pbar.update(1)

            with open(TEST_RESULT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "Mode": f"{TEST_MODE}, ime_type: {ime_type}, error_rate: {error_rate_str}",
                        "Total Test Cases": len(test_data),
                        "Accuracy": correct / len(test_data),
                        "Avg. MRR": sum(mrr_scores) / len(mrr_scores),
                        "Avg. Time Spend": sum(time_spends) / len(time_spends),
                        # "Logs": logs,
                    },
                    f,
                    ensure_ascii=False,
                )
