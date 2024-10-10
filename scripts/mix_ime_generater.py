import os
import random

from tqdm import tqdm

from data_preprocess.keystroke_converter import KeyStrokeConverter
from data_preprocess.typo_generater import TypoGenerater
from multiprocessing import Pool


SPLITS = "\t©©©\t"
MAX_DATA_LINE = 20000
CONVERT_LANGUAGES = ["bopomofo", "cangjie", "pinyin", "english"]
ERROR_RATE = 0.1

def process_line(chinese_line, english_line, k_num):
    two_languages = random.sample(CONVERT_LANGUAGES, k=k_num)
    line_keystroke, line_answer = "", ""
    for language in two_languages:
        keystroke = ""
        if language == "english":
            keystroke += KeyStrokeConverter.convert(english_line, convert_type=language)
        else:
            keystroke += KeyStrokeConverter.convert(chinese_line, convert_type=language)
        
        keystroke = TypoGenerater.generate(keystroke, error_type="random", error_rate=ERROR_RATE)

        line_keystroke += keystroke
        line_answer += f"(\"{language}\", \"{keystroke}\")"

    return line_keystroke + SPLITS + line_answer


if __name__ == "__main__":
    PLAIN_TEXT_DATA_PATH = ".\\Plain_Text_Datasets\\"
    TARGET_PATH = "..\\..\\System_Test\\labled_mix_ime_r{}.txt".format(str(ERROR_RATE).replace(".","-"))
    files = ["wlen1-3_cc100.txt", "wlen1-3_English_multi.txt"]

    random.seed(32)

    chinese_lines = []
    english_lines = []
    with open(os.path.join(PLAIN_TEXT_DATA_PATH, files[0]), "r", encoding="utf-8") as f:
        chinese_lines = (line.strip() for line in f)
        chinese_lines = random.sample(list(chinese_lines), MAX_DATA_LINE)

    with open(os.path.join(PLAIN_TEXT_DATA_PATH, files[1]), "r", encoding="utf-8") as f:
        english_lines = (line.strip() for line in f)
        english_lines = random.sample(list(english_lines), MAX_DATA_LINE)

    MAX_LINES = min(len(chinese_lines), len(english_lines), MAX_DATA_LINE)
    chinese_lines = chinese_lines[:MAX_LINES]
    english_lines = english_lines[:MAX_LINES]
    k_num_list = [1, 2] * (MAX_LINES // 2) + [1] * (MAX_LINES % 2)

    outlines = []
    mix_count = {"1": k_num_list.count(1), "2": k_num_list.count(2)}

    
    with tqdm(total=MAX_LINES) as pbar:
        with Pool() as pool:
            results = []
            for chinese_line, english_line, k_num in zip(chinese_lines, english_lines, k_num_list):
                results.append(pool.apply_async(process_line, (chinese_line, english_line, k_num)))

            for result in results:
                outlines.append(result.get())
                pbar.update(1)

    outlines.insert(0, f"# Total: {MAX_LINES}, 1: {mix_count['1']}, 2: {mix_count['2']}")
    with open(TARGET_PATH, "w", encoding="utf-8") as f:
        for outline in outlines:
            f.write(outline.strip() + "\n")
