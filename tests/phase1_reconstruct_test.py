import os
import random
import json
from tqdm import tqdm
from datetime import datetime


from multilingual_ime.key_event_handler import KeyEventHandler
from multilingual_ime.ime import PINYIN_IME, BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME


random.seed(42)
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")
IMES = [BOPOMOFO_IME, CANGJIE_IME, PINYIN_IME, ENGLISH_IME]


def error_rate_to_string(error_rate: float) -> str:
    if error_rate == 0:
        return "0"
    return "r" + str(error_rate).replace(".", "-")


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


def list_get(list: list, item: any, default: any):
    try:
        return list[item]
    except IndexError:
        return default


if __name__ == "__main__":
    # Test Parameters
    TEST_CASES_NUMBER = 10
    TEST_ERROR_RATE = 0
    NUM_OF_TOKEN = 5
    IME_CHANGE_MODE = True
    NUM_OF_MIX_IME = 3
    # Settings
    # No settings

    if NUM_OF_MIX_IME > NUM_OF_TOKEN:
        raise ValueError("NUM_OF_MIX_IME should be less than or equal to NUM_OF_TOKEN")

    error_rate_str = error_rate_to_string(TEST_ERROR_RATE)
    TEST_RESULT_JSON_PATH = f"reports\\phase1_reconstruct_n{NUM_OF_TOKEN}-imes_{error_rate_str}_test_result_{TIMESTAMP}.json"

    BOPOMOFO_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_bopomofo_{error_rate_str}_test.txt"
    )
    CANGJIE_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_cangjie_{error_rate_str}_test.txt"
    )
    PINYIN_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_pinyin_{error_rate_str}_test.txt"
    )
    ENGLISH_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_english_{error_rate_str}_test.txt"
    )

    print("Loading Test Tokens")
    max_token_numbers = TEST_CASES_NUMBER * NUM_OF_TOKEN * 10
    bopomofo_tokens = load_tokens(
        BOPOMOFO_TOKEN_TEST_DATA_PATH, max_size=max_token_numbers
    )
    cangjie_tokens = load_tokens(
        CANGJIE_TOKEN_TEST_DATA_PATH, max_size=max_token_numbers
    )
    pinyin_tokens = load_tokens(PINYIN_TOKEN_TEST_DATA_PATH, max_size=max_token_numbers)
    english_tokens = load_tokens(
        ENGLISH_TOKEN_TEST_DATA_PATH, max_size=max_token_numbers
    )

    def generate_tokens(ime_list: list[str]) -> list[str]:
        tokens = []
        prev_ime_type = None
        for ime_type in ime_list:
            if ime_type == BOPOMOFO_IME:
                tokens.append(random.choice(bopomofo_tokens))
            elif ime_type == CANGJIE_IME:
                tokens.append(random.choice(cangjie_tokens))
            elif ime_type == PINYIN_IME:
                tokens.append(random.choice(pinyin_tokens))
            elif ime_type == ENGLISH_IME:
                if prev_ime_type == ENGLISH_IME:
                    tokens.append(" ")
                    tokens.append(random.choice(english_tokens))
                else:
                    tokens.append(random.choice(english_tokens))
            prev_ime_type = ime_type
        return tokens

    print("Generating Test Data")
    test_data = []
    for _ in range(TEST_CASES_NUMBER):

        if IME_CHANGE_MODE:  # Define number of IME change
            sub_ime_types = random.choices(IMES, k=NUM_OF_MIX_IME)
            random.shuffle(sub_ime_types)
            duplicate = random.choices(
                sub_ime_types, k=(NUM_OF_TOKEN - len(sub_ime_types))
            )

            ime_list = []
            for ime_type in sub_ime_types:
                if ime_type in duplicate:
                    ime_list.extend([ime_type, ime_type])
                    duplicate.remove(ime_type)
                else:
                    ime_list.append(ime_type)

            print("------")
            print(sub_ime_types)
            print(duplicate)
            assert len(ime_list) == NUM_OF_TOKEN

            tokens = generate_tokens(ime_list)
            test_x = "".join(tokens)
            test_label = tokens
            test_data.append((test_x, test_label))
        else:  # Count by number of token
            ime_list = random.sample(IMES, NUM_OF_MIX_IME)

            tokens = generate_tokens(ime_list)
            test_x = "".join(tokens)
            test_label = tokens
            test_data.append((test_x, test_label))

    print("Testing Reconstruct")
    with tqdm(total=len(test_data), desc=f"Testing IME Reconstruct {ime_type}") as pbar:
        correct = 0
        mrr_scores = []  # Mean Reciprocal Rank
        wrong_logs = []

        test_key_event_handler = KeyEventHandler()

        for test_x, y_label in test_data:
            y_pred = test_key_event_handler.phase_1(test_x)
            # print("y_pred", y_pred)
            # print("y_label", y_label)
            if y_label in y_pred:
                correct += 1
                mrr_scores.append(1 / (y_pred.index(y_label) + 1))
            else:
                mrr_scores.append(0)
                wrong_logs.append(
                    {"Input": test_x, "Output": y_pred, "Correnct": y_label}
                )

            pbar.update()

    with open(TEST_RESULT_JSON_PATH, "w") as f:
        json.dump(
            {
                "Accuracy": correct / len(test_data),
                "Avg.MRR": sum(mrr_scores) / len(mrr_scores),
                "MRR": mrr_scores,
                "WrongLog": wrong_logs,
            },
            f,
        )
