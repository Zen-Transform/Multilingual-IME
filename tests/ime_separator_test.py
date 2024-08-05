import os
import random
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm


from multilingual_ime.data_preprocess.typo_generater import TypoGenerater
from multilingual_ime.data_preprocess.keystroke_converter import KeyStrokeConverter
from multilingual_ime.ime_separator import IMESeparator
from multilingual_ime.core.multi_job_processing import multiprocessing

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
        keystroke = KeyStrokeConverter.convert(text, convert_type=ime_type)
        mix_ime_keystrokes += TypoGenerater.generate(
            keystroke, error_type=ERROR_TYPE, error_rate=ERROR_RATE
        )
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
        + f"Output_Len: {len(separat_result)}\n",
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
            "Mix_count": {
                "mix_1": num_of_mix_ime_list.count(1),
                "mix_2": num_of_mix_ime_list.count(2),
            },
        }

        with tqdm(total=MAX_DATA_LINE) as pbar:
            with Pool() as pool:

                def updete_pbar(*a):
                    pbar.update()

                reuslt = []
                for num_of_mix_ime in num_of_mix_ime_list:
                    sampled_languages = random.sample(
                        CONVERT_LANGUAGES, k=num_of_mix_ime
                    )
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

    def test_separator():
        if os.path.exists(TEST_FILE_PATH):
            if (
                input(
                    f"File {TEST_FILE_PATH} already exists, do you want to overwrite it? (y/n): "
                )
                == "y"
            ):
                os.remove(TEST_FILE_PATH)
                generate_mix_ime_test_data()
        else:
            generate_mix_ime_test_data()

        assert os.path.exists(TEST_FILE_PATH), f"{TEST_FILE_PATH} not found"

        # Testing separator on mixed ime data
        separator = IMESeparator(
            use_cuda=False
        )  # use_cuda=False for multi-processing test

        wrong_answer_logs = []
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
                separat_result = separator.separate(mix_ime_keystrokes)
                results.append(
                    mutiprocess_test(separator, mix_ime_keystrokes, label_answer)
                )

            total_test_example = len(results)
            correct_count = 0
            prediction_len_count = 0
            prediction_len_count_correct = 0
            len_score = 0
            for result in results:
                if result["Correct"]:
                    correct_count += 1
                    prediction_len_count_correct += result["Output_Len"]
                else:
                    wrong_answer_logs.append(result["Test_log"])

                numerater = 1 if result["Correct"] else 0
                denumerator = result["Output_Len"]
                len_score += numerater / denumerator if denumerator > 0 else 0
                prediction_len_count += denumerator

        except KeyboardInterrupt:
            print("User interrupt")

        finally:
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
                    + f"Len Score: {len_score/total_test_example}\n\n"
                    + f"\n".join(wrong_answer_logs)
                )

    test_separator()
