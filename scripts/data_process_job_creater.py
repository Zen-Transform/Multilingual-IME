import os
import json

# Create V2 Dataset

if __name__ == "__main__":
    PLAIN_TEXT_DATASET_PATH = ".\\Datasets\\Plain_Text_Datasets\\"
    KEY_STROKE_DATASET_PATH = ".\\Datasets\\Key_Stroke_Datasets\\"
    TRAIN_DATASET_PATH = ".\\Datasets\\Train_Datasets\\"
    TEST_DATASET_PATH = ".\\Datasets\\Test_Datasets\\"
    FILE_NAME = ".\\scripts\\data_process_job.json"

    job_list = []

    # Create Cleaned Dataset from existing plain text dataset
    mode = "clean"
    group = [("chinese", "Chinese_WebCrawlData_cc100.txt"), ("english", "English_multi_news.txt")]
    skip_clean = False

    for language, src_file in group:
        job_list.append({
            "mode": mode,
            "description": f"Clean {src_file} to {language}",
            "input_file_path": PLAIN_TEXT_DATASET_PATH + src_file, 
            "output_file_path": PLAIN_TEXT_DATASET_PATH + f"{src_file.replace('.txt', '-ch.txt')}",
            "language": language,
            "status": "done" if skip_clean else None
            })
    

    # Split Dataset into words
    mode = "split_word"
    group = [("ch", "Chinese_WebCrawlData_cc100-ch.txt"), ("en", "English_multi_news-ch.txt")]
    min_split_word_len = 1
    max_split_word_len = 3
    skip_split_word = False

    output_file_name_list = []
    for language, src_file in group:
        dataset_name = None
        if src_file.find("WebCrawlData_cc100") >= 0:
            dataset_name = "cc100"
        elif src_file.find("English_multi_news") >= 0:
            dataset_name = "English_multi"
        elif src_file.find("Chinese_news") >= 0:
            dataset_name = "Chinese_news"
        elif src_file.find("gossip") >= 0:
            dataset_name = "gossip"
        else:
            raise ValueError("Invalid file name: " + src_file)

        output_file_name = f"wlen{min_split_word_len}-{max_split_word_len}_{dataset_name}.txt"
        output_file_name_list.append(output_file_name)
        job_list.append({
            "mode": mode,
            "description": f"Split {src_file} into words with length {min_split_word_len}-{max_split_word_len}",
            "input_file_path": PLAIN_TEXT_DATASET_PATH + src_file, 
            "output_file_path": PLAIN_TEXT_DATASET_PATH + output_file_name,
            "min_split_word_len": min_split_word_len,
            "max_split_word_len": max_split_word_len,
            "language": language,
            "status": "done" if skip_split_word else None
            })

        
    # Split Plain Text Dataset into Train and Test Dataset
    mode = "split"
    src_files = ["wlen1-3_cc100.txt", "wlen1-3_English_multi.txt"]
    train_test_split_ratio = 0.5
    skip_split = False

    for src_file in src_files:
        train_file_name = src_file.replace(".txt", "_train.txt")
        test_file_name = src_file.replace(".txt", "_test.txt")

        job_list.append({
            "mode": mode,
            "description": f"Split {src_file} with {train_test_split_ratio} into {train_file_name} and {test_file_name} datasets.",
            "input_file_path": PLAIN_TEXT_DATASET_PATH + src_file,
            "train_file_path": PLAIN_TEXT_DATASET_PATH + train_file_name,
            "test_file_path": PLAIN_TEXT_DATASET_PATH + test_file_name,
            "train_test_split_ratio": train_test_split_ratio,
            "status": "done" if skip_split else None
            })


    # Create KeyStroke Dataset from existing plain text dataset
    mode = "convert"
    src_files = ["wlen1-3_cc100_train.txt", "wlen1-3_cc100_test.txt"]
    convert_types = ["bopomofo", "cangjie", "pinyin"]
    skip_convert = False

    def is_train_file(file_name):
        if "_train" in file_name:
            return True
        elif "_test" in file_name:
            return False
        else:
            raise ValueError("Invalid file name: " + file_name + " , must contain '_train' or '_test'")


    convert_files = []
    for src_file in src_files:
        for convert_type in convert_types:

            output_file_name = "0_{}_{}".format(convert_type, src_file.replace("_train", '').replace("_test", ''))
            output_file_path = (TRAIN_DATASET_PATH if is_train_file(src_file) else TEST_DATASET_PATH) + output_file_name
            convert_files.append(output_file_path)
            job_list.append({
                "mode": mode,
                "description": f"Convert {src_file} to {convert_type}",
                "input_file_path": PLAIN_TEXT_DATASET_PATH + src_file, 
                "output_file_path": output_file_path,
                "convert_type": convert_type,
                "status": "done" if skip_convert else None
                })

    mode = "convert"
    src_files = ["wlen1-3_English_multi_train.txt", "wlen1-3_English_multi_test.txt"]  # todo: fix this
    convert_types = ["english"]
    for src_file in src_files:
        for convert_type in convert_types:

            output_file_name = "0_{}_{}".format(convert_type, src_file.replace("_train", '').replace("_test", ''))
            output_file_path = (TRAIN_DATASET_PATH if is_train_file(src_file) else TEST_DATASET_PATH) + output_file_name
            convert_files.append(output_file_path)
            job_list.append({
                "mode": mode,
                "description": f"Convert {src_file} to {convert_type}",
                "input_file_path": PLAIN_TEXT_DATASET_PATH + src_file, 
                "output_file_path": output_file_path,
                "convert_type": convert_type,
                "status": "done" if skip_convert else None
                })

    # Create Error Dataset from existing keystroke dataset
    mode = "gen_error"
    src_files = convert_files
    error_rates = [0.1, 0.05]
    error_types = ["random", "8adjacency"]
    skip_gen_error = False


    with_error_files = []
    for src_file in src_files:
        for error_type in error_types:
            for error_rate in error_rates:
                error_rate_name = str(error_rate).replace(".", "-")
                error_type_name = "r" if error_type == "random" else "8a" if error_type == "8adjacency" else error_type

                src_file_name = os.path.basename(src_file).split("_")[1:]
                output_file_name = f"{error_type_name}{error_rate_name}_{'_'.join(src_file_name)}"
                with_error_files.append(os.path.dirname(src_file) + "\\" + output_file_name)
                
                job_list.append({
                    "mode": mode,
                    "description": f"Generate '{error_type}' error for {src_file} with error rate {error_rate}",
                    "input_file_path": src_file, 
                    "output_file_path": os.path.dirname(src_file) + "\\" + output_file_name,
                    "error_type": error_type,
                    "error_rate": error_rate,
                    "status": "done" if skip_gen_error else None
                    })

    


    # merge all files and label them
    mode = "merge"
    target_languages = ["bopomofo", "cangjie", "pinyin", "english"]
    prefixes = ["0", "r0-1"]
    skip_merge = False

    for target_language in target_languages:
        for prefix in prefixes:
            job_list.append({
                "mode": mode,
                "description": f"Merge all {prefix} files in training datasets and label them with {target_language}",
                "prefix": prefix,
                "input_file_paths": TRAIN_DATASET_PATH,
                "output_file_path": TRAIN_DATASET_PATH + f"labeled_{target_language}_{prefix}_train.txt",
                "language": target_language,
                "status": "done" if skip_merge else None
                })
            job_list.append({
                "mode": mode,
                "description": f"Merge all {prefix} files in testing datasets and label them with {target_language}",
                "prefix": prefix,
                "input_file_paths": TEST_DATASET_PATH,
                "output_file_path": TEST_DATASET_PATH + f"labeled_{target_language}_{prefix}_test.txt",
                "language": target_language,
                "status": "done" if skip_merge else None
                })

    if os.path.exists(FILE_NAME):
        if input("Do you want to append the new job to the existing job file? (y/n)") == "y":
            print("Appending the new job to the existing job file")
            old_job_list = json.load(open(FILE_NAME, "r"))
            des = [job["description"] for job in old_job_list]

            for job in job_list:
                if job["description"] not in des:
                    old_job_list.append(job)
            
            with open(FILE_NAME, "w") as f:
                json.dump(old_job_list, f, indent=4)
        elif input("Do you want to overwrite the existing job file? (y/n)") == "y":
            print("Overwriting the existing job file")
            with open(FILE_NAME, "w") as f:
                json.dump(job_list, f, indent=4)
        else:
            print("Exiting...")
    else:
        with open(FILE_NAME, "w") as f:
            json.dump(job_list, f, indent=4)