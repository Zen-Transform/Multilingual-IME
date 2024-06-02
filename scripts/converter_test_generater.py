import os
import random
import math
from multiprocessing import Pool

from tqdm import tqdm

from multilingual_ime.data_preprocess.keystroke_converter import KeyStrokeConverter
from multilingual_ime.data_preprocess.typo_generater import TypoGenerater

def process_line(language, line):
    keystroke = KeyStrokeConverter.convert(line, convert_type=language)
    keystroke = TypoGenerater.generate(keystroke, error_rate=0.05, error_type="random")
    return f"{keystroke}\t{language}\t{line}"


SPLITS = "\t"
MAX_DATA_LINE = 20000
CONVERT_LANGUAGES = ["bopomofo", "cangjie", "pinyin", "english"]


if __name__ == "__main__":
    PLAIN_TEXT_DATA_PATH = ".\\Plain_Text_Datasets\\"
    TARGET_PATH = "..\\..\\System_Test\\converter_test_r0-05.txt"
    files = ["wlen1-3_cc100.txt", "wlen1-3_English_multi.txt"]

    random.seed(32)
    
    outlines = []
    language_count = {"bopomofo": 0, "cangjie": 0, "pinyin": 0, "english": 0}
    job_list = []
    with open(os.path.join(PLAIN_TEXT_DATA_PATH, files[0]), "r", encoding="utf-8") as f:
        chinese_lines = []
        chinese_lines = (line.strip() for line in f)
        chinese_lines = [line for line in chinese_lines if len(line) == 1]
        chinese_lines = random.sample(list(chinese_lines), math.floor(MAX_DATA_LINE * 3/4))

        for i, line in enumerate(chinese_lines):
            if i % 3 == 0:
                job_list.append(("bopomofo", line))
                language_count["bopomofo"] += 1
            elif i % 3 == 1:
                job_list.append(("cangjie", line))
                language_count["cangjie"] += 1
            else:
                job_list.append(("pinyin", line))
                language_count["pinyin"] += 1

    with open(os.path.join(PLAIN_TEXT_DATA_PATH, files[1]), "r", encoding="utf-8") as f:
        english_lines = []
        english_lines = (line.strip() for line in f)
        english_lines = [line for line in english_lines if len(line.split(" ")) == 1]
        english_lines = [line for line in english_lines if line != ""]
        english_lines = [line for line in english_lines if line.isalpha()]
        english_lines = random.sample(list(english_lines), math.floor(MAX_DATA_LINE * 1/4))
        for i, line in enumerate(english_lines):
            job_list.append(("english", line))
            language_count["english"] += 1


    with tqdm(total=len(job_list)) as pbar:
        with Pool() as pool:
            results = []
            for job in job_list:
                results.append(pool.apply_async(process_line, job))

            for result in results:
                outlines.append(result.get())
                pbar.update(1)

    outlines.insert(0, f"{language_count}")
    with open(TARGET_PATH, "w", encoding="utf-8") as f:
        for outline in outlines:
            f.write(outline.strip() + "\n")
