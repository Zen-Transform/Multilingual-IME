import json
import os
import random

from tqdm import tqdm

from multilingual_ime.data_preprocess.language_cleaner import LanguageCleaner
from multilingual_ime.data_preprocess.keystroke_converter import KeyStrokeConverter
from multilingual_ime.data_preprocess.typo_generater import TypoGenerater
from multilingual_ime.data_preprocess.ime_keys import IMEKeys

random.seed(42)

def split_train_test_file(input_file_path, train_file_path, test_file_path, train_test_split_size):
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        split_index = int(len(lines) * train_test_split_size)
        train_lines = lines[:split_index]
        test_lines = lines[split_index:]

    with open(train_file_path, "w", encoding="utf-8") as file:
        for line in train_lines:
            file.write(line)

    with open(test_file_path, "w", encoding="utf-8") as file:
        for line in test_lines:
            file.write(line)

def split_by_word(input_file_path, output_file_path, min_split_word_len, max_split_word_len, language="en"):
    with open(input_file_path, "r", encoding="utf-8") as file_in:
        lines = file_in.readlines()
        output_lines = []
        if language == "en":
            input_string = "".join(lines)
            words = [word for line in input_string.split('\n') for word in line.split(' ')]
            i = 0
            with tqdm(total=len(words)) as pbar:
                while i < len(words):
                    output_lines.append(' '.join(words[i:i+1]))
                    output_lines.append(' '.join(words[i+1:i+3]))
                    output_lines.append(' '.join(words[i+3:i+6]))
                    pbar.update(6)
                    i += 6
        elif language == "ch":
            joined_lines = "".join(lines).replace("\n", "")
            words = [word for word in joined_lines]
            i = 0
            with tqdm(total=len(words)) as pbar:
                while i < len(words):
                    output_lines.append(''.join(words[i:i+1]))
                    output_lines.append(''.join(words[i+1:i+3]))
                    output_lines.append(''.join(words[i+3:i+6]))
                    pbar.update(6)
                    i += 6
        else:
            raise ValueError("Invalid language: " + language)
    
    with open(output_file_path, "w", encoding="utf-8") as file_out:
        file_out.write("\n".join(output_lines))

def cut_keystroke_by(input_file_path, output_file_path, cut_out_len):
    with open(input_file_path, "r", encoding="utf-8") as file_in:
        keystroke_lines = file_in.readlines()
        keystroke_lines = [line.replace("\n", "") for line in keystroke_lines if line.strip() != ""]
        joined_keystroke_lines = "".join(keystroke_lines)
        out_keystrokes = [joined_keystroke_lines[i:i+cut_out_len] for i in range(0, len(joined_keystroke_lines), cut_out_len)]
    with open(output_file_path, "w", encoding="utf-8") as file_out:
        file_out.write("\n".join(out_keystrokes))

def merge_files(input_file_paths, output_file_path, target_language, prefix=""):
    files = os.listdir(input_file_paths)
    src_file_names = [file for file in files if file.startswith(f"{prefix}_")]
    assert len(src_file_names) == 4

    for file_name in tqdm(src_file_names, desc="Processing files"):
        with open(input_file_paths + file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
            lines = [line.replace("\n", "") for line in lines] # remove the \n in the line
            new_lines = [line + "\t1" if target_language in file_name else line + "\t0" for line in lines]
        
        with open(os.path.join(output_file_path), "a", encoding="utf-8") as file:
            for line in new_lines:
                file.write(line + "\n")

def write_list_to_file(file_path: str, lines: list) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")


ADVERSARIAL_CHAR_NUM = 1
def generate_adversarial_file(input_file_path: str, output_file_path:str, language: str) -> None:
    def add_adversarial_keystroke(keystroke: str, language: str) -> str:
        if language == "english":
            adversarial_keys = IMEKeys.universal_keys - IMEKeys.english_keys
        elif language == "bopomofo":
            adversarial_keys = IMEKeys.universal_keys - IMEKeys.bopomofo_keys
        elif language == "cangjie":
            adversarial_keys = IMEKeys.universal_keys - IMEKeys.cangjie_keys
        elif language == "pinyin":
            adversarial_keys = IMEKeys.universal_keys - IMEKeys.pinyin_keys
        else:
            raise ValueError("Invalid language: " + language)

        adversarial_chars =  random.sample(list(adversarial_keys), ADVERSARIAL_CHAR_NUM)
        for adversarial_char in adversarial_chars:
            index = random.randint(0, len(keystroke))
            keystroke = keystroke[:index] + adversarial_char + keystroke[index:]        
        return keystroke

    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [line.replace("\n", "") for line in lines]
        lines = [line for line in lines if line.split("\t")[1] == "1"]
    outlines = []
    for line in lines:
        outlines.append("{}\t0".format(add_adversarial_keystroke(line.split("\t")[0], language)))
    write_list_to_file(output_file_path, outlines)


REDUNDANT_CHAR_NUM = 1
def generate_redundant_file(input_file_path: str, output_file_path:str, language: str) -> None:
    def add_redundant_keystroke(keystroke: str, language: str) -> str:
        if language == "english":
            redundant_keys = IMEKeys.english_keys
        elif language == "bopomofo":
            redundant_keys = IMEKeys.bopomofo_keys
        elif language == "cangjie":
            redundant_keys = IMEKeys.cangjie_keys
        elif language == "pinyin":
            redundant_keys = IMEKeys.pinyin_keys
        else:
            raise ValueError("Invalid language: " + language)

        redundant_chars =  random.sample(list(redundant_keys), REDUNDANT_CHAR_NUM)
        keystroke = keystroke + "".join(redundant_chars)       
        return keystroke

    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [line.replace("\n", "") for line in lines]
        lines = [line for line in lines if line.split("\t")[1] == "1"]
    outlines = []
    for line in lines:
        outlines.append("{}\t0".format(add_redundant_keystroke(line.split("\t")[0], language)))
    write_list_to_file(output_file_path, outlines)


if __name__ == "__main__":
    NUM_PROCESSES = 4
    PROCESS_JOB_FILE = ".\\scripts\\data_process_job.json"

    KEYSTROKE_DATASET_PATH = ".\\Datasets\\KeyStroke_Datasets\\"
    TRAIN_DATASET_PATH = ".\\Datasets\\Train_Datasets\\"
    TEST_DATASET_PATH = ".\\Datasets\\Test_Datasets\\"

    job_list = json.load(open(PROCESS_JOB_FILE, "r"))


    # clear all files in the dataset folders
    exiting_train_files = [file for file in os.listdir(TRAIN_DATASET_PATH) if file.startswith("labeled_")]
    exiting_test_files = [file for file in os.listdir(TEST_DATASET_PATH) if file.startswith("labeled_")]
    if len(exiting_train_files) > 0:
        if input("Do you want to remake all files in Train_Datasets? (y/n)") == "y":
            print("Removing all labeled files in Key_Stroke_Datasets")
            for file_name in exiting_train_files:
                os.remove(TRAIN_DATASET_PATH + file_name)

    if len(exiting_test_files) > 0:
        if input("Do you want to remake all files in Test_Datasets? (y/n)") == "y":
            print("Removing all labeled files in Key_Stroke_Datasets")
            for file_name in exiting_test_files:
                os.remove(TEST_DATASET_PATH + file_name)

    unfinished_jobs = []
    for job in job_list:
        if job.get("status") != "done" or job.get("status") is None:
            try: 
                if job["mode"] == "clean":
                    LanguageCleaner.clean_file_parallel(job["input_file_path"], job["output_file_path"], job["language"], num_processes=NUM_PROCESSES)
                elif job["mode"] == "convert":
                    KeyStrokeConverter.convert_file_parallel(job["input_file_path"], job["output_file_path"], job["convert_type"], num_processes=NUM_PROCESSES)
                elif job["mode"] == "gen_error":
                    TypoGenerater.generate_file_parallel(job["input_file_path"], job["output_file_path"], job["error_type"], job["error_rate"], num_processes=NUM_PROCESSES)
                elif job["mode"] == "split":
                    split_train_test_file(job["input_file_path"], job["train_file_path"], job["test_file_path"], job["train_test_split_ratio"])
                elif job["mode"] == "split_word":
                    split_by_word(job["input_file_path"], job["output_file_path"], job["min_split_word_len"], job["max_split_word_len"], job["language"])
                elif job["mode"] == "cut_keystroke":
                    cut_keystroke_by(job["input_file_path"], job["output_file_path"], job["cut_out_len"])
                elif job["mode"] == "merge":
                    merge_files(job["input_file_paths"], job["output_file_path"], job["language"], job["prefix"])
                elif job["mode"] == "gen_reverse":
                    generate_adversarial_file(job["input_file_path"], job["output_file_path"], job["language"])
                elif job["mode"] == "gen_redundant":
                    generate_redundant_file(job["input_file_path"], job["output_file_path"], job["language"])
                else:
                    raise ValueError("Invalid mode: " + job["mode"])
                
                print(f"Success: In {job['mode']}, {job['description']}")
                job["status"] = "done"
            except Exception as e:
                print(f"Error: In {job['mode']}, {job['description']}")
                print("Error: " + str(e))
                unfinished_jobs.append(job)
                job["status"] = "error"
                continue
            

    if len(unfinished_jobs) > 0:
        print("----- Unfinished jobs -----")
        for job in unfinished_jobs:
            print(job)

    with open(PROCESS_JOB_FILE, "w") as f:
        json.dump(job_list, f, indent=4)