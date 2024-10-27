import os
import random
import re
import ast
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm

from data_preprocess.keystroke_converter import KeyStrokeConverter
from data_preprocess.typo_generater import TypoGenerater
from multilingual_ime.ime_handler import decide_tokenizer, IMEHandler

random.seed(42)
# Generate config
DATA_AND_LABEL_SPLITTER = "\t©©©\t"
USER_DEFINE_MAX_DATA_LINE = 20000
USER_DEFINE_MAX_TEST_LINE = 2000
CONVERT_LANGUAGES = ["bopomofo", "cangjie", "pinyin", "english"]
ERROR_TYPE = "random"
ERROR_RATE = 0
NUM_OF_MIX_IME = 2
MIX_WITH_DIFFERENT_NUM_OF_IME = False


# others
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")

# File path
CHINESE_PLAIN_TEXT_FILE_PATH = (
    ".\\Datasets\\Plain_Text_Datasets\\wlen1-3_cc100_test.txt"
)
ENGLISH_PLAIN_TEXT_FILE_PATH = (
    ".\\Datasets\\Plain_Text_Datasets\\wlen1-3_English_multi_test.txt"
)
TEST_FILE_PATH = ".\\tests\\test_data\\labeled_mix_ime_{}{}.txt".format(
    "r" if ERROR_TYPE == "random" else ("8a" if ERROR_TYPE == "8adjacency" else "e"),
    str(ERROR_RATE).replace(".", "-"),
)
SEPARATOR_TEST_RESULT_FILE_PATH = (
    f".\\reports\\ime_separator_test_result_{TIMESTAMP}.txt"
)
SEPARATOR_CONVERTOR_TEST_RESULT_FILE_PATH = (
    f".\\reports\\ime_separator_converter_test_result_{TIMESTAMP}.txt"
)


def process_generate_line(mix_ime_group: list[(str, str)]) -> str:
    mix_ime_keystrokes, answer_group, plain_texts, tokens = "", [], "", []
    for ime_type, text in mix_ime_group:

        ime_keystrokes = ""
        if ime_type == "english":
            ime_keystrokes = TypoGenerater.generate(
                KeyStrokeConverter.convert(text, convert_type=ime_type),
                error_type=ERROR_TYPE,
                error_rate=ERROR_RATE,
            )
            tokens.extend(decide_tokenizer(ime_type, ime_keystrokes))

        else:
            for word in text:
                word_keystroke = TypoGenerater.generate(
                    KeyStrokeConverter.convert(word, convert_type=ime_type),
                    error_type=ERROR_TYPE,
                    error_rate=ERROR_RATE,
                )
                ime_keystrokes += word_keystroke
                tokens.append(word_keystroke)

        mix_ime_keystrokes += ime_keystrokes
        answer_group.append(f'("{ime_type}", "{ime_keystrokes}")')
        plain_texts += text

    return (
        mix_ime_keystrokes
        + DATA_AND_LABEL_SPLITTER
        + ",".join(answer_group)
        + DATA_AND_LABEL_SPLITTER
        + plain_texts
        + DATA_AND_LABEL_SPLITTER
        + str(tokens)
    )


def generate_mix_ime_test_data():
    chinese_lines = []
    english_lines = []
    with open(CHINESE_PLAIN_TEXT_FILE_PATH, "r", encoding="utf-8") as f:
        chinese_lines = [line.strip() for line in f.readlines()]
        chinese_lines = [line for line in chinese_lines if len(line) > 0]
    with open(ENGLISH_PLAIN_TEXT_FILE_PATH, "r", encoding="utf-8") as f:
        english_lines = [line.strip() for line in f.readlines()]
        english_lines = [line for line in english_lines if len(line) > 0]

    MAX_DATA_LINE = min(
        len(chinese_lines), len(english_lines), USER_DEFINE_MAX_DATA_LINE
    )
    chinese_lines = random.sample(chinese_lines, MAX_DATA_LINE * 3)
    english_lines = random.sample(english_lines, MAX_DATA_LINE)
    print(f"Generating {MAX_DATA_LINE} lines of mixed language data")

    if MIX_WITH_DIFFERENT_NUM_OF_IME:
        num_of_mix_ime_list = (
            [x for x in range(1, NUM_OF_MIX_IME + 1)]
            * ((MAX_DATA_LINE // NUM_OF_MIX_IME) + 1)
        )[:MAX_DATA_LINE]
    else:
        num_of_mix_ime_list = [NUM_OF_MIX_IME] * MAX_DATA_LINE
    assert (
        len(num_of_mix_ime_list) == MAX_DATA_LINE
    ), "error in num_of_mix_ime_list length"

    config = {
        "Data_format": "mix_ime_keystrokes"
        + DATA_AND_LABEL_SPLITTER
        + "separate_answer"
        + DATA_AND_LABEL_SPLITTER
        + "plain_text",
        "Total_lines": MAX_DATA_LINE,
        "NUM_OF_MIX_IME": NUM_OF_MIX_IME,
        "ERROR_RATE": ERROR_RATE,
        "Mix_count": {  # Fix: mix could larger than 2
            "mix_1": num_of_mix_ime_list.count(1),
            "mix_2": num_of_mix_ime_list.count(2),
        },
    }

    with tqdm(total=MAX_DATA_LINE) as pbar, Pool() as pool:

        def update_pbar(*a):
            pbar.update()

        result = []
        for num_of_mix_ime in num_of_mix_ime_list:
            sampled_languages = random.sample(CONVERT_LANGUAGES, k=num_of_mix_ime)
            mix_ime_group = []
            for language in sampled_languages:
                if language == "english":
                    mix_ime_group.append((language, english_lines.pop(0)))
                else:
                    mix_ime_group.append((language, chinese_lines.pop(0)))

            result.append(
                pool.apply_async(
                    process_generate_line, args=(mix_ime_group,), callback=update_pbar
                )
            )

        result = [res.get() for res in result]
        result.insert(0, str(config))
        with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(result))


ime_handler = IMEHandler()


def ime_handle_line_test(line: str):
    mix_ime_keystrokes, answer_group, plain_texts, label_tokens = line.split(
        DATA_AND_LABEL_SPLITTER
    )
    start_time = datetime.now()
    candidate_sentences = ime_handler.get_candidate_sentences(mix_ime_keystrokes)
    time_spend = (datetime.now() - start_time).total_seconds()

    correct = False
    for sentence in candidate_sentences:
        if sentence["sentence"] == ast.literal_eval(label_tokens):
            correct = True
            break
    

    return {
        "Plain_text": plain_texts,
        "Correct": correct,
        "time spend": time_spend,
        "Test_log": f"Plain_text: {plain_texts}\nKeystroke: {mix_ime_keystrokes}\nCorrect: {correct}\nTime spend: {time_spend}\n",
    }


def ime_handler_perf_test():
    # ime_handler = IMEHandler()
    # if not os.path.exists(TEST_FILE_PATH) or (
    #     os.path.exists(TEST_FILE_PATH) and _user_want_to_overwrite(TEST_FILE_PATH)
    # ):
    #     generate_mix_ime_test_data()
    assert os.path.exists(TEST_FILE_PATH), f"{TEST_FILE_PATH} not found"

    with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        test_config, test_lines = eval(lines[0]), lines[1:]
        test_lines = random.sample(
            test_lines, min(len(test_lines), USER_DEFINE_MAX_TEST_LINE)
        )

    try:
        results = []
        for line in tqdm(test_lines):
            results.append(ime_handle_line_test(line))

    except KeyboardInterrupt:
        print("User interrupt")

    finally:
        total_test_example = len(results)
        correct_count = sum(1 for result in results if result["Correct"])

        print("============= Test Result =============")
        print(f"{test_config}")
        print(f"Test Date: {TIMESTAMP}")
        print(f"Total Test Sample: {total_test_example}")
        print(
            f"Accuracy: {correct_count/total_test_example}, {correct_count}/{total_test_example}"
        )
        with open(
            SEPARATOR_CONVERTOR_TEST_RESULT_FILE_PATH, "w", encoding="utf-8"
        ) as f:
            f.write(
                f"============= Test Result =============\n"
                + f"{test_config}\n"
                + f"Test Date: {TIMESTAMP}\n"
                + f"Total Test Sample: {total_test_example}\n"
                + f"Correct: {correct_count}\n"
                + f"Accuracy: {correct_count/total_test_example}, {correct_count}/{total_test_example}\n"
                + f"\n"
                + f"\n".join([result["Test_log"] for result in results])
            )


if __name__ == "__main__":
    # generate_mix_ime_test_data()
    ime_handler_perf_test()
    # print(process_line([("bopomofo", "你好"), ("english", "hello")]))
    # print(process_line([("english", "promise"),("bopomofo", "院不")]))
    # print(process_line([("english", "Amanda"),("pinyin", "丈夫第")]))
    # print(process_line([("pinyin", "出"),("cangjie", "的聖")]))
