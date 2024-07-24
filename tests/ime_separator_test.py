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
CONVERT_LANGUAGES = ["bopomofo", "cangjie", "pinyin", "english"]
ERROR_TYPE = "random"
ERROR_RATE = 0
NUM_OF_MIX_IME = 2
MIX_WITH_DIFFERENT_NUM_OF_IME = False

# others
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")

# File path
CHINESE_PLAIN_TEXT_FILE_PATH = ".\\Datasets\\Plain_Text_Datasets\\wlen1-3_cc100_test.txt"
ENGLISH_PLAIN_TEXT_FILE_PATH = ".\\Datasets\\Plain_Text_Datasets\\wlen1-3_English_multi_test.txt"
TEST_FILE_PATH = ".\\tests\\test_data\\labeld_mix_ime_{}{}.txt".format( \
    "r" if ERROR_TYPE == "random" else ("8a" if ERROR_TYPE == "8adjacency" else "e"), \
    str(ERROR_RATE).replace(".", "-"),\
    )
TEST_RESULT_FILE_PATH = f".\\reports\\ime_separator_test_result_{TIMESTAMP}.txt"


def process_line(chinese_line: str, english_line: str, k_num: int):
    sampled_languages = random.sample(CONVERT_LANGUAGES, k=k_num)
    mix_ime_keystrokes, line_answer = "", ""
    for language in sampled_languages:
        keystroke = ""
        if language == "english":
            keystroke += KeyStrokeConverter.convert(english_line, convert_type=language)
        else:
            keystroke += KeyStrokeConverter.convert(chinese_line, convert_type=language)
        
        keystroke = TypoGenerater.generate(keystroke, error_type="random", error_rate=ERROR_RATE)

        mix_ime_keystrokes += keystroke
        line_answer += f"(\"{language}\", \"{keystroke}\")"

    return mix_ime_keystrokes + DATA_AND_LABEL_SPLITTER + line_answer


def mutiprocess_test(separator: IMESeparator, mix_ime_keystrokes: str, separat_answer: str) -> dict:
    separat_result = separator.separate(mix_ime_keystrokes)
    is_correct = separat_answer in separat_result
    return {
        "Correct": is_correct,
        "Test_log": 
            f"Mix_ime: {mix_ime_keystrokes}\n" + \
            f"Answer: {separat_answer}\n" + \
            f"Result: {separat_result}\n"
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
        
        MAX_DATA_LINE = min(len(chinese_lines), len(english_lines), USER_DEFINE_MAX_DATA_LINE)
        chinese_lines = random.sample(chinese_lines, MAX_DATA_LINE)
        english_lines = random.sample(english_lines, MAX_DATA_LINE)
        print(f"Generating {MAX_DATA_LINE} lines of mixed language data")

        if MIX_WITH_DIFFERENT_NUM_OF_IME:
            num_of_mix_ime_list = ([x for x in range(1, NUM_OF_MIX_IME + 1)] * ((MAX_DATA_LINE // NUM_OF_MIX_IME) + 1))[:MAX_DATA_LINE]
        else:
            num_of_mix_ime_list = [NUM_OF_MIX_IME] * MAX_DATA_LINE
        assert len(num_of_mix_ime_list) == MAX_DATA_LINE, "error in num_of_mix_ime_list length"
        
        config = {
            "total_lines": MAX_DATA_LINE,
            "NUM_OF_MIX_IME": NUM_OF_MIX_IME,
            "ERROR_RATE": ERROR_RATE,
            "mix_count": {
                "mix_1": num_of_mix_ime_list.count(1),
                "mix_2": num_of_mix_ime_list.count(2)
            }
        }

        with tqdm(total=MAX_DATA_LINE) as pbar:
            with Pool() as pool:
                def updete_pbar(*a):
                    pbar.update()

                reuslt = []
                for chinese_line, english_line, num_of_mix_ime in zip(chinese_lines, english_lines, num_of_mix_ime_list):
                    reuslt.append(pool.apply_async(process_line, args=(chinese_line, english_line, num_of_mix_ime), callback=updete_pbar))

                reuslt = [res.get() for res in reuslt]
                reuslt.insert(0, str(config))
                with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
                    f.write("\n".join(reuslt))


    if os.path.exists(TEST_FILE_PATH):
        if input(f"File {TEST_FILE_PATH} already exists, do you want to overwrite it? (y/n): ") == "y":
            os.remove(TEST_FILE_PATH)
            generate_mix_ime_test_data()
    else:
        generate_mix_ime_test_data()

    assert os.path.exists(TEST_FILE_PATH), f"{TEST_FILE_PATH} not found"


    # Testing mix ime
    separator = IMESeparator(use_cuda=True)  # use_cuda=False for multi-processing test


    wrong_answers = []
    with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        test_config, test_lines = eval(lines[0]), lines[1:]

    try:
        job_args = []
        for line in test_lines:
            mix_ime_keystrokes, line_answer = line.strip().split(DATA_AND_LABEL_SPLITTER)
            separat_answer = eval("["+line_answer.replace(")(", "), (")+"]")
            job_args.append((separator, mix_ime_keystrokes, separat_answer))

        multiprocessing_result = multiprocessing(mutiprocess_test, job_args, show_progress=True, preserve_order=True)
        total_count = len(multiprocessing_result)
        correct_count = sum([result["Correct"] for result in multiprocessing_result])
        for result in multiprocessing_result:
            if not result["Correct"]:
                wrong_answers.append(result["Test_log"])

    except KeyboardInterrupt:
        print("User interrupt")
    
    finally: 
        print("============= Test Result =============")       
        print(f"{test_config}")
        print(f"correct: {correct_count}")
        print(f"total: {total_count}")
        print(f"Accuracy: {correct_count/total_count}, {correct_count}/{total_count}")
        with open(TEST_RESULT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(
                f"============= Test Result =============\n" + \
                f"{test_config}\n" + \
                f"Test Date: {TIMESTAMP}\n" + \
                f"correct: {correct_count}\n" + \
                f"total: {total_count}\n" + \
                f"Accuracy: {correct_count/total_count}, {correct_count}/{total_count}\n\n" + \
                f"\n".join(wrong_answers))
