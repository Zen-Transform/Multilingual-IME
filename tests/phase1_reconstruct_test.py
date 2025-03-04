import os
from datetime import datetime
import random
import json

from tqdm import tqdm

from data_preprocess.typo_generater import TypoGenerater
from multilingual_ime.key_event_handler import KeyEventHandler
from multilingual_ime.ime import PINYIN_IME, BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME


random.seed(42)
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")


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


def get_token_efficient(ime_type: str, error_rate: int) -> str:
    while True:
        error_rate_str = error_rate_to_string(error_rate)
        bopomofo_token_test_data_path = (
            f"Datasets\\Test_Datasets\\labeled_wlen1_bopomofo_{error_rate_str}_test.txt"
        )
        cangjie_token_test_data_path = (
            f"Datasets\\Test_Datasets\\labeled_wlen1_cangjie_{error_rate_str}_test.txt"
        )
        pinyin_token_test_datat_path = (
            f"Datasets\\Test_Datasets\\labeled_wlen1_pinyin_{error_rate_str}_test.txt"
        )
        english_token_test_data_path = (
            f"Datasets\\Test_Datasets\\labeled_wlen1_english_{error_rate_str}_test.txt"
        )

        if ime_type == BOPOMOFO_IME:
            line = get_random_line_efficient(bopomofo_token_test_data_path)
        elif ime_type == CANGJIE_IME:
            line = get_random_line_efficient(cangjie_token_test_data_path)
        elif ime_type == PINYIN_IME:
            line = get_random_line_efficient(pinyin_token_test_datat_path)
        elif ime_type == ENGLISH_IME:
            line = get_random_line_efficient(english_token_test_data_path)
        else:
            raise ValueError(f"IME {ime_type} is not supported")

        items = line.strip().split("\t")

        if len(items) == 2 and items[1] == "1" and len(items[0]) != 0:
            return items[0]


def generate_test_case(
    mix_imes: list[str], n_token: int, error_rate: int, one_time_change: bool
) -> tuple:
    # Generate IME list
    if one_time_change:
        mix_imes = mix_imes.copy()
        random.shuffle(mix_imes)
        duplicate = random.choices(mix_imes, k=n_token - len(mix_imes))

        ime_list = []
        for ime_type in mix_imes:
            ime_list.extend([ime_type] * (duplicate.count(ime_type) + 1))

        assert len(ime_list) == n_token
    else:
        ime_list = random.choices(mix_imes, k=n_token)

    # Generate tokens
    tokens = []
    prev_ime_type = None
    for ime_type in ime_list:
        if prev_ime_type == ENGLISH_IME:
            tokens.append(" ")

        token = get_token_efficient(ime_type=ime_type, error_rate=error_rate)
        tokens.append(
            TypoGenerater.generate(token, error_type="random", error_rate=error_rate)
        )

        prev_ime_type = ime_type
    return ("".join(tokens), tokens, ime_list)


if __name__ == "__main__":
    # Test Parameters
    TEST_CASES_NUMBER = 1000
    TEST_ERROR_RATES = [0, 0.1, 0.05]
    NUM_OF_TOKENS = [7, 6, 5, 4]
    MIX_IMES = [PINYIN_IME, ENGLISH_IME]
    ONE_TIME_CHANGE_MODE = True

    # Settings
    # No settings

    for test_error_rate in TEST_ERROR_RATES:
        for num_of_token in NUM_OF_TOKENS:
            if ONE_TIME_CHANGE_MODE and len(MIX_IMES) > num_of_token:
                raise ValueError(
                    f"Number of MIX_IME: {len(MIX_IMES)} should be less than or equal to NUM_OF_TOKEN: {num_of_token}"
                )

            ERROR_RATE_STR = error_rate_to_string(test_error_rate)
            MIX_IME_STR = "-".join(MIX_IMES)
            TEST_RESULT_JSON_PATH = f"reports\\phase1_reconstruct_test_oneTime-{ONE_TIME_CHANGE_MODE}_{MIX_IME_STR}_{num_of_token}-tokens_{ERROR_RATE_STR}_{TIMESTAMP}.json"

            test_key_event_handler = KeyEventHandler()
            with tqdm(total=TEST_CASES_NUMBER, desc="Testing IME Reconstruct") as pbar:
                corrects = []
                mrr_scores = []  # Mean Reciprocal Rank
                # wrong_logs = []
                time_spends = []
                infinite_counts = []

                for i in range(TEST_CASES_NUMBER):

                    test_x, y_label, imes = generate_test_case(
                        mix_imes=MIX_IMES,
                        n_token=num_of_token,
                        error_rate=test_error_rate,
                        one_time_change=ONE_TIME_CHANGE_MODE,
                    )
                    start_time = datetime.now()
                    y_pred = test_key_event_handler.new_reconstruct(test_x)
                    time_spend = datetime.now() - start_time

                    total_distance = (
                        test_key_event_handler._calculate_sentence_distance(y_pred[0])
                    )

                    if total_distance == float("inf"):
                        infinite_counts.append(1)
                    else:
                        infinite_counts.append(0)

                    # print("test_x", test_x)
                    # print("y_pred", y_pred)
                    # print("y_label", y_label)

                    if y_label in y_pred:
                        corrects.append(1)
                        mrr_scores.append(1 / (y_pred.index(y_label) + 1))
                    else:
                        corrects.append(0)
                        mrr_scores.append(0)
                        # wrong_logs.append(
                        #     {
                        #         "Input": test_x,
                        #         "Output": y_pred,
                        #         "Correnct": y_label,
                        #         "IME": imes,
                        #     }
                        # )

                    time_spends.append(time_spend.total_seconds())
                    pbar.update()

            print(
                "------------------------------------------------------------\n"
                + f"Test Result IME Reconstruct: n{num_of_token}\n"
                + f"Testing IME Reconstruct: n{num_of_token}\n"
                + f"One Time Change Mode: {ONE_TIME_CHANGE_MODE}\n"
                + f"Mix IME: {MIX_IMES}\n"
                + f"Error Rate: {test_error_rate}\n"
                + f"Total Test Cases: {len(corrects)}\n"
                + f"ACC: {sum(corrects) / len(corrects)}\n"
                + f"Time: {sum(time_spends) / len(time_spends)}\n"
                + f"Mean Reciprocal Rank: {sum(mrr_scores) / len(mrr_scores)}\n"
            )
            assert (
                len(corrects) == TEST_CASES_NUMBER
            ), f"Mismatch on length of {len(corrects)} and test case number {TEST_CASES_NUMBER}"
            with open(TEST_RESULT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "Mode": f"Reconstruct, one_time_change_mode: {ONE_TIME_CHANGE_MODE},"
                        f"mix_ime: {MIX_IMES}, num_of_token: {num_of_token},  error_rate: {test_error_rate}",
                        "Total Test Cases": len(corrects),
                        "Accuracy": sum(corrects) / len(corrects),
                        "Avg.MRR": sum(mrr_scores) / len(mrr_scores),
                        "Avg.TimeSpend": sum(time_spends) / len(time_spends),
                        "MRR": mrr_scores,
                        "TimeSpend": time_spends,
                        "Infinite Count": infinite_counts,
                        "Total Infinite Count": sum(infinite_counts),
                        "Infinite Rate": sum(infinite_counts) / len(infinite_counts),
                        # "WrongLog": wrong_logs,
                    },
                    f,
                )
