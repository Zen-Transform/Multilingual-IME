import argparse
import re
import multiprocessing

from tqdm import tqdm


class LanguageCleaner:
    @staticmethod
    def cleanChinese(input_string:str, reserve_puncutation_symbol: bool = False, reserve_fullwidth_symbol: bool = False, reserve_new_line: bool = True) -> str:
        """
        Clean the input string to only contain Chinese characters and punctuation
        but not the newline

        Args:
            input_string (str): The input string

        Returns:
            str: The cleaned string
        """
        reg_reserve_exp = r"\u4e00-\u9fff"
        if reserve_puncutation_symbol:
            reg_reserve_exp += r"\u3000-\u303f"
        if reserve_fullwidth_symbol:
            reg_reserve_exp += r"\uff00-\uffef"
        if reserve_new_line:
            reg_reserve_exp += r"\n"

        return re.sub(f"[^{reg_reserve_exp}]", "", input_string)

    @staticmethod
    def cleanEnglish(input_string:str, reserve_punctuation: bool = False, reserve_numbers: bool = False, reserve_newline: bool = True) -> str:
        """
        Clean the input string to only contain English characters, punctuation and numbers

        Args:
            input_string (str): The input string

        Returns:
            str: The cleaned string
        """
        reg_reserve_exp = r"a-zA-Z "
        if reserve_punctuation:
            reg_reserve_exp += r"\.,\?!"
        if reserve_numbers:
            reg_reserve_exp += r"0-9"
        if reserve_newline:
            reg_reserve_exp += r"\n"

        return re.sub(f"[^{reg_reserve_exp}]", "", input_string)

    @staticmethod
    def clean(input_string:str, language:str, reserve_newline:bool) -> str:
        """
        Clean the input string to only contain the specified language characters and punctuation

        Args:
            input_string (str): The input string
            language (str): The language to reserve, "chinese" or "english"
            reserve_newline (bool): Whether to reserve the newline character

        Returns:
            str: The cleaned string
        """

        if language == "chinese":  # fix: bad code, hard coded reserve settings
            return LanguageCleaner.cleanChinese(input_string, reserve_puncutation_symbol=False, reserve_fullwidth_symbol=False, reserve_new_line=reserve_newline)
        elif language == "english":
            return LanguageCleaner.cleanEnglish(input_string, reserve_punctuation=False, reserve_numbers=False, reserve_newline=reserve_newline)
        else:
            raise ValueError("Error: language '{}' is not supported".format(language))


    def clean_file(input_file_path:str, output_file_path: str, language:str, reserve_newline:bool=True):
        """
        Clean the input file and write the cleaned content to the output file

        Args:
            input_file_path (str): The input file path
            output_file_path (str): The output file path
            language (str): The language to reserve, "chinese" or "english"
            reserve_newline (bool): Whether to reserve the newline character
        """
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.read()
            cleaned_data = LanguageCleaner.clean(data, language, reserve_newline)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_data)

    @staticmethod
    def _clean_file_chunk(input_queue, output_queue, language:str, reserve_newline:bool=True):
        while True:
            chunk_index, chunk = input_queue.get()
            if chunk is None:
                output_queue.put(None)  # Signal the main process that this worker is done
                break
            cleaned_chunk = LanguageCleaner.clean(chunk, language=language, reserve_newline=reserve_newline)
            output_queue.put((chunk_index, cleaned_chunk))


    def clean_file_parallel(input_file_path:str, output_file_path:str, language:str, reserve_newline:bool=True, num_processes:int=4): # fix: make it cleaner
        """
        Clean the input file and write the cleaned content to the output file in parallel

        Args:
            input_file_path (str): The input file path
            output_file_path (str): The output file path
            language (str): The language to reserve, "chinese" or "english"
            reserve_newline (bool): Whether to reserve the newline character
            chuck_job: The job to be done on the chunks
            num_processes (int, optional): The number of processes to use. Defaults to 4.
        """
        chunk_size = 10000
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()


        # Read the input file and split it into chunks
        with open(input_file_path, 'r', encoding='utf-8') as f:
            chunk_index = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                input_queue.put((chunk_index, chunk))
                chunk_index += 1

        # Add termination signals to the input queue
        for _ in range(num_processes):
            input_queue.put((None, None))

        # Process chunks in parallel
        processes = []
        for _ in range(num_processes):
            p = multiprocessing.Process(target=LanguageCleaner._clean_file_chunk, args=(input_queue, output_queue, language, reserve_newline))
            processes.append(p)
            p.start()

        # Collect and reorder cleaned chunks
        cleaned_chunks = [None] * chunk_index
        workers_done = 0
        with tqdm(total=chunk_index) as pbar:
            while workers_done < num_processes:
                try:
                    chunk_info = output_queue.get(timeout=1)  # Timeout to prevent hanging
                    if chunk_info is None:
                        workers_done += 1
                    else:
                        chunk_index, cleaned_chunk = chunk_info
                        cleaned_chunks[chunk_index] = cleaned_chunk
                        pbar.update(1)  # Update progress bar for each processed chunk
                except multiprocessing.TimeoutError:
                    # Timeout occurred, check if processes are still alive
                    alive_processes = [p.is_alive() for p in processes]
                    if not any(alive_processes):
                        break  # All processes have terminated

        # Terminate any remaining processes
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Write reordered cleaned chunks to the output file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for cleaned_chunk in cleaned_chunks:
                if cleaned_chunk is not None:
                    f.write(cleaned_chunk)
        print(f"Cleaning Success: {output_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean the input file to only contain the specified language characters and punctuation")
    parser.add_argument("language", 
                        choices=["chinese", "english"],
                        help="The language to reserve, 'chinese' or 'english'")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f", "--file", 
        help="The input file path")
    group.add_argument(
        "-i", "--input",
        help="The input string")

    parser.add_argument(
        "-o", "--output", 
        help="The output file path")
    parser.add_argument("--reserve_newline",
                        default=True,
                        action="store_true",
                        help="Whether to reserve the newline character")

    args = parser.parse_args()
    if args.file:
        LanguageCleaner.clean_file(args.file, args.output, args.language, args.reserve_newline)
    elif args.string:
        cleaned_string = LanguageCleaner.clean(args.string, args.language, args.reserve_newline)
        print(cleaned_string)

if __name__ == '__main__':
    # main()

    test_input_ch = "★ 內建智慧晶片可自動，切換和雙系統接上即可使用\n你超棒"
    test_input_en = "★ This is a test string for English ，cleaning\nYou are awesome!!!"
    print("===== Before cleaning =====")
    print(test_input_ch)
    print(test_input_en)
    print("\n===== After cleaning =====")
    print(LanguageCleaner.clean(test_input_ch, "chinese", reserve_newline=True))
    print(LanguageCleaner.clean(test_input_en, "english", reserve_newline=True))
    # dir_path = os.path.dirname(__file__)
    # input_file = os.path.abspath(os.path.join(dir_path, "..\\Plain_Text_Datasets\\Chinese_news.txt"))
    # print(input_file)
    # output_file = os.path.abspath(os.path.join(dir_path, "..\\Plain_Text_Datasets\\Chinese_news-ch.txt"))
    # LanguageCleaner.clean_file_parallel(input_file, output_file, language="chinese", reserve_newline=True)
