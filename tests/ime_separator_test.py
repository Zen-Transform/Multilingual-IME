import os
import random
import re
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm


from multilingual_ime.data_preprocess.typo_generater import TypoGenerater
from multilingual_ime.data_preprocess.keystroke_converter import KeyStrokeConverter
from multilingual_ime.ime_separator import IMESeparator
from multilingual_ime.ime_handler import IMEHandler
from multilingual_ime.core.multi_job_processing import multiprocessing

random.seed(42)
# Generate config
DATA_AND_LABEL_SPLITTER = "\t©©©\t"
USER_DEFINE_MAX_DATA_LINE = 20000
USER_DEFINE_MAX_TEST_LINE = 2000
CONVERT_LANGUAGES = ["bopomofo", "cangjie", "pinyin", "english"]
ERROR_TYPE = "random"
ERROR_RATE = 0.1
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
TEST_FILE_PATH = ".\\tests\\test_data\\labeld_mix_ime_{}{}.txt".format(
    "r" if ERROR_TYPE == "random" else ("8a" if ERROR_TYPE == "8adjacency" else "e"),
    str(ERROR_RATE).replace(".", "-"),
)
SEPARATOR_TEST_RESULT_FILE_PATH = (
    f".\\reports\\ime_separator_test_result_{TIMESTAMP}.txt"
)
SEPARATOR_CONVERTOR_TEST_RESULT_FILE_PATH = (
    f".\\reports\\ime_separator_converter_test_result_{TIMESTAMP}.txt"
)


def process_line(mix_ime_group: list[(str, str)]) -> str:
    mix_ime_keystrokes, answer_group, plain_texts = "", [], ""
    for ime_type, text in mix_ime_group:
        keystroke = TypoGenerater.generate(
            KeyStrokeConverter.convert(text, convert_type=ime_type),
            error_type=ERROR_TYPE,
            error_rate=ERROR_RATE,
        )
        mix_ime_keystrokes += keystroke
        answer_group.append(f'("{ime_type}", "{keystroke}")')
        plain_texts += text

    return (
        mix_ime_keystrokes
        + DATA_AND_LABEL_SPLITTER
        + ",".join(answer_group)
        + DATA_AND_LABEL_SPLITTER
        + plain_texts
    )


def mutiprocess_test(
    separator: IMESeparator, mix_ime_keystrokes: str, separate_answer: list
) -> dict:
    separat_result = separator.separate(mix_ime_keystrokes)
    is_correct = separate_answer in separat_result
    return {
        "Correct": is_correct,
        "Output_Len": len(separat_result),
        "Test_log": f"Input: {mix_ime_keystrokes}\n"
        + f"Label: {separate_answer}\n"
        + f"Output: {separat_result}\n"
        + f"Output_Len: {len(separat_result)}\n"
        + f"Correct: {is_correct}\n",
    }


def _separate_characters_and_words(text):
    # Regular expression to match Chinese characters and English words
    pattern = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+")
    # Find all matches in the text
    matches = pattern.findall(text)
    return matches


def batch_sep_conv_test(
    ime_handler: IMEHandler, mix_ime_keystrokes: str, separate_answer: list
) -> dict:
    sentence_candidate_suggestions = ime_handler.get_candidate(mix_ime_keystrokes)
    is_correct = True
    if len(separate_answer) != len(sentence_candidate_suggestions):
        is_correct = False
    for word_candidates, answer in zip(sentence_candidate_suggestions, separate_answer):
        if answer not in word_candidates:
            is_correct = False
            break
    return {
        "Correct": is_correct,
        "Test_log": f"Input: {mix_ime_keystrokes}\n"
        + f"Label: {separate_answer}\n"
        + f"Output: {sentence_candidate_suggestions}\n"
        + f"Correct: {is_correct}\n",
    }


if __name__ == "__main__":

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

            def updete_pbar(*a):
                pbar.update()

            reuslt = []
            for num_of_mix_ime in num_of_mix_ime_list:
                sampled_languages = random.sample(CONVERT_LANGUAGES, k=num_of_mix_ime)
                mix_ime_group = []
                for language in sampled_languages:
                    if language == "english":
                        mix_ime_group.append((language, english_lines.pop(0)))
                    else:
                        mix_ime_group.append((language, chinese_lines.pop(0)))

                reuslt.append(
                    pool.apply_async(
                        process_line, args=(mix_ime_group,), callback=updete_pbar
                    )
                )

            reuslt = [res.get() for res in reuslt]
            reuslt.insert(0, str(config))
            with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(reuslt))

    def _user_want_to_overwrite(file_path: str) -> bool:
        return (
            input(
                f"File {file_path} already exists, do you want to overwrite it? (y/n): "
            )
            == "y"
        )

    def _calculate_len_score(results: dict) -> float:
        len_score = 0
        for result in results:
            numerater = 1 if result["Correct"] else 0
            denumerator = result["Output_Len"]
            len_score += numerater / denumerator if denumerator > 0 else 0
        return len_score

    def test_separator():  # Testing separator on mixed ime data
        if not os.path.exists(TEST_FILE_PATH) or (
            os.path.exists(TEST_FILE_PATH) and _user_want_to_overwrite(TEST_FILE_PATH)
        ):
            generate_mix_ime_test_data()

        assert os.path.exists(TEST_FILE_PATH), f"{TEST_FILE_PATH} not found"

        separator = IMESeparator(
            use_cuda=False
        )  # use_cuda=False for multi-processing test

        with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            test_config, test_lines = eval(lines[0]), lines[1:]
            test_lines = random.sample(
                test_lines, min(len(test_lines), USER_DEFINE_MAX_TEST_LINE)
            )
        try:
            results = []
            for line in tqdm(test_lines):
                mix_ime_keystrokes, line_answer = line.strip().split(
                    DATA_AND_LABEL_SPLITTER
                )[:2]
                label_answer = eval("[" + line_answer.replace(")(", "), (") + "]")
                results.append(
                    mutiprocess_test(separator, mix_ime_keystrokes, label_answer)
                )

        except KeyboardInterrupt:
            print("User interrupt")

        finally:
            total_test_example = len(results)
            correct_count = sum(1 for result in results if result["Correct"])
            prediction_len_count = sum(result["Output_Len"] for result in results)
            prediction_len_count_correct = sum(
                result["Output_Len"] for result in results if result["Correct"]
            )
            len_score = _calculate_len_score(results)

            print("============= Test Result =============")
            print(f"{test_config}")
            print(
                f"Accuracy: {correct_count/total_test_example}, {correct_count}/{total_test_example}"
            )
            print(f"Len Score: {len_score/total_test_example}")
            with open(SEPARATOR_TEST_RESULT_FILE_PATH, "w", encoding="utf-8") as f:
                f.write(
                    f"============= Test Result =============\n"
                    + f"{test_config}\n"
                    + f"Test Date: {TIMESTAMP}\n"
                    + f"Total Test Sample: {total_test_example}\n"
                    + f"Correct: {correct_count}\n"
                    + f"Total Predictions: {prediction_len_count}\n"
                    + f"Average Output Len: {prediction_len_count/total_test_example}\n"
                    + f"Average Correct Output Len: {prediction_len_count_correct/total_test_example}\n"
                    + f"Accuracy: {correct_count/total_test_example}, {correct_count}/{total_test_example}\n"
                    + f"Len Score: {len_score/total_test_example}\n"
                    + f"\n"
                    + f"\n".join([result["Test_log"] for result in results])
                )

    def test_separator_conveter():  # Testing separator & converter on mixed ime data
        ime_handler = IMEHandler()
        if not os.path.exists(TEST_FILE_PATH) or (
            os.path.exists(TEST_FILE_PATH) and _user_want_to_overwrite(TEST_FILE_PATH)
        ):
            generate_mix_ime_test_data()
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
                mix_ime_keystrokes, line_plain_text_answer = line.strip().split(
                    DATA_AND_LABEL_SPLITTER
                )[0::2]

                results.append(
                    batch_sep_conv_test(
                        ime_handler,
                        mix_ime_keystrokes,
                        _separate_characters_and_words(line_plain_text_answer),
                    )
                )

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

    test_separator()
    # test_separator_conveter()
