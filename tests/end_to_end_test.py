import os
import random
from datetime import datetime

import jieba
from tqdm import trange
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from multilingual_ime.ime import (
    BOPOMOFO_IME,
    ENGLISH_IME,
    CANGJIE_IME,
    PINYIN_IME,
    JAPANESE_IME,
)
from multilingual_ime.key_event_handler import KeyEventHandler
from data_preprocess.typo_generater import TypoGenerater
from data_preprocess.keystroke_converter import KeyStrokeConverter


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


def get_chinese_word(n: int) -> list[str]:
    CHINESE_PLAIN_TEXT_FILE_PATH = (
        ".\\Datasets\\Plain_Text_Datasets\\Chinese_WebCrawlData_cc100-ch.txt"
    )
    jieba_words = []
    while len(jieba_words) <= n:
        line = get_random_line_efficient(CHINESE_PLAIN_TEXT_FILE_PATH)
        jieba_words = jieba.lcut(line)
        jieba_words = [word for word in jieba_words if len(word) > 0]
    assert len(jieba_words) > 0, f"Got empty line from {CHINESE_PLAIN_TEXT_FILE_PATH}"

    words = ""
    while True:
        new_word = jieba_words[random.randint(0, len(jieba_words) - 1)]
        if len(words + new_word) == n:
            words += new_word
            break
        elif len(words + new_word) < n:
            words += new_word
        else:
            words += new_word[: n - len(words)]
            break

    assert len(words) == n
    return list(words)


def get_english_word(n: int) -> list[str]:
    ENGLISH_PLAIN_TEXT_FILE_PATH = (
        ".\\Datasets\\Plain_Text_Datasets\\English_multi_news-ch.txt"
    )
    line = get_random_line_efficient(ENGLISH_PLAIN_TEXT_FILE_PATH)
    words = line.split(" ")
    words = [word for word in words if len(word) > 0]
    return random.choices(words, k=n)


def generate_test_case(
    mix_imes: list[str], n_token: int, error_rate: int, one_time_change: bool
) -> tuple:
    if one_time_change:
        mix_imes = mix_imes.copy()
        random.shuffle(mix_imes)
        if n_token < len(mix_imes):
            ime_list = mix_imes[:n_token]
        elif n_token == len(mix_imes):
            ime_list = mix_imes
        else:
            duplicate = random.choices(mix_imes, k=n_token - len(mix_imes))

            ime_list = []
            for currentime_type in mix_imes:
                ime_list.extend([currentime_type] * (duplicate.count(currentime_type) + 1))

        assert len(ime_list) == n_token
    else:
        ime_list = random.choices(mix_imes, k=n_token)

    # Generate tokens
    words = []
    i = 0
    while i < len(ime_list):
        currentime_type = ime_list[i]
        n = 1
        while i < len(ime_list) - 1 and ime_list[i + 1] == currentime_type:
            n += 1
            i += 1

        if currentime_type in [BOPOMOFO_IME, CANGJIE_IME, PINYIN_IME]:
            words.extend(get_chinese_word(n))
        elif currentime_type == ENGLISH_IME:
            words.extend(get_english_word(n))
        else:
            raise ValueError(f"IME {currentime_type} is not supported")
        i += 1

    assert len(words) == n_token

    tokens = []
    new_words = []
    prev_ime_type = None
    for word, ime in zip(words, ime_list):
        if prev_ime_type == ENGLISH_IME:
            tokens.append(" ")
            new_words.append(" ")
        tokens.append(
            TypoGenerater.generate(
                KeyStrokeConverter.convert(word, convert_type=ime),
                error_type="random",
                error_rate=error_rate,
            )
        )
        new_words.append(word)
        prev_ime_type = ime

    new_words = [new_word for new_word in new_words if new_word != ""]
    return ("".join(tokens), tokens, new_words, ime_list)


def list_to_safe_string(l: list[str]) -> str:
    total_string = '"['
    for i, item in enumerate(l):
        if isinstance(item, str):
            total_string += "'" + item.replace("'", "\\'").replace('"', '\\"')
            total_string += "'," if i < len(l) - 1 else "'"
    total_string += ']"'
    return total_string


random.seed(32)
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")


if __name__ == "__main__":
    # Test Parameters
    TEST_CASE_NUMBER = 500
    MIX_IMES = [CANGJIE_IME, ENGLISH_IME]
    TEST_ERROR_RATES = [0, 0.05, 0.1]
    NUM_OF_TOKENS = [7, 6, 5, 4]
    ONE_TIME_CHANGE_MODE = True

    test_ime_handler = KeyEventHandler()
    test_ime_handler.set_activation_status(
        BOPOMOFO_IME, True if BOPOMOFO_IME in MIX_IMES else False
    )
    test_ime_handler.set_activation_status(
        ENGLISH_IME, True if ENGLISH_IME in MIX_IMES else False
    )
    test_ime_handler.set_activation_status(
        CANGJIE_IME, True if CANGJIE_IME in MIX_IMES else False
    )
    test_ime_handler.set_activation_status(
        PINYIN_IME, True if PINYIN_IME in MIX_IMES else False
    )
    test_ime_handler.set_activation_status(
        JAPANESE_IME, True if JAPANESE_IME in MIX_IMES else False
    )

    for test_error_rate in TEST_ERROR_RATES:
        for num_of_token in NUM_OF_TOKENS:
            ERROR_RATE_STR = error_rate_to_string(test_error_rate)
            MIX_IME_STR = "-".join(MIX_IMES)
            TEST_RESULT_CSV_PATH = f"reports\\end2end_test_oneTime-{ONE_TIME_CHANGE_MODE}_{MIX_IME_STR}_{num_of_token}-tokens_{ERROR_RATE_STR}_{TIMESTAMP}.csv"
            with open(TEST_RESULT_CSV_PATH, "a", encoding="utf-8") as f:
                f.write(
                    "e2e_correct,sep_correct,BLEUn4,BLEUn3,BLEUn2,BLEUn1,unigram,bigrams,trigrams,fourgrams,time_spend,x_test_str,y_pred_sentence,y_label_tokens,y_pred_tokens,y_pred_sep,y_label_sep\n"
                )
            method1_smoothing_function = SmoothingFunction().method1

            for _ in trange(
                TEST_CASE_NUMBER,
                desc=f"Testing End-to-End {MIX_IME_STR}, n{num_of_token}, e{test_error_rate}",
            ):
                x_test_str, y_label_sep, y_label_tokens, ime_list = generate_test_case(
                    mix_imes=MIX_IMES,
                    n_token=num_of_token,
                    error_rate=test_error_rate,
                    one_time_change=ONE_TIME_CHANGE_MODE,
                )

                start_time = datetime.now()
                y_pred_tokens = test_ime_handler.end_to_end(x_test_str)
                y_pred_sentence = "".join(y_pred_tokens)
                time_spend = (datetime.now() - start_time).total_seconds()
                y_pred_sep = test_ime_handler.new_reconstruct(x_test_str)[0]
                e2e_correct = (
                    True if y_pred_sentence == "".join(y_label_tokens) else False
                )
                sep_correct = True if y_pred_sep == y_label_sep else False
                BLEUn4 = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    smoothing_function=method1_smoothing_function,
                )
                BLEUn3 = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    weights=(1, 1, 1, 0),
                    smoothing_function=method1_smoothing_function,
                )
                BLEUn2 = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    weights=(1, 1, 0, 0),
                    smoothing_function=method1_smoothing_function,
                )
                BLEUn1 = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    weights=(1, 0, 0, 0),
                    smoothing_function=method1_smoothing_function,
                )
                unigram = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    weights=(1, 0, 0, 0),
                    smoothing_function=method1_smoothing_function,
                )
                bigrams = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    weights=(0, 1, 0, 0),
                    smoothing_function=method1_smoothing_function,
                )
                trigrams = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    weights=(0, 0, 1, 0),
                    smoothing_function=method1_smoothing_function,
                )
                fourgrams = sentence_bleu(
                    [y_label_tokens],
                    y_pred_tokens,
                    weights=(0, 0, 0, 1),
                    smoothing_function=method1_smoothing_function,
                )
                with open(TEST_RESULT_CSV_PATH, "a", encoding="utf-8") as f:
                    f.write(
                        f'{e2e_correct},{sep_correct},{BLEUn4},{BLEUn3},{BLEUn2},{BLEUn1},{unigram},{bigrams},{trigrams},{fourgrams},{time_spend},"{x_test_str}","{y_pred_sentence}",{list_to_safe_string(y_label_tokens)},{list_to_safe_string(y_pred_tokens)},{list_to_safe_string(y_label_sep)},{list_to_safe_string(y_pred_sep)}\n'
                    )
