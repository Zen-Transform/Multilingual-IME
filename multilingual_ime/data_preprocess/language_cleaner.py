import os
import warnings

import re
import multiprocessing
from tqdm import tqdm

def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"\033[91m"+ "Call to deprecated function {func.__name__}" + "\033[0m", category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

# Set warnings to raise an error by default
warnings.simplefilter('error', DeprecationWarning)

class LanguageCleaner:
    @staticmethod
    @deprecated
    def cleanChinese(input_string:str) -> str:
        """
        Clean the input string to only contain Chinese characters and punctuation
        but not the newline

        Args:
            input_string (str): The input string

        Returns:
            str: The cleaned string
        """

        return re.sub(r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\n]", "", input_string)

    @staticmethod
    @deprecated
    def cleanEnglish(input_string:str) -> str:
        """
        Clean the input string to only contain English characters, punctuation and numbers

        Args:
            input_string (str): The input string

        Returns:
            str: The cleaned string
        """
        return re.sub(r"[^a-zA-Z0-9\.,\?! ]", "", input_string)

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

        if language == "chinese":  # fix: make newline regex more readable, separate the logic
            pattern = r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\n]" if reserve_newline else r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"
        elif language == "english":
            pattern = r"[^a-zA-Z0-9\.,\?! \n]" if reserve_newline else r"[^a-zA-Z0-9\.,\?! ]"
        else:
            raise ValueError("Error: language '{}' is not supported".format(language))

        return re.sub(pattern, "", input_string)

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

if __name__ == '__main__':
    # test_input = "★ 內建智慧晶片可自動切換和雙系統接上即可使用"
    # print(LanguageCleaner.cleanChinese(test_input))
    dir_path = os.path.dirname(__file__)
    input_file = os.path.abspath(os.path.join(dir_path, "..\\Plain_Text_Datasets\\Chinese_news.txt"))
    print(input_file)
    output_file = os.path.abspath(os.path.join(dir_path, "..\\Plain_Text_Datasets\\Chinese_news-ch.txt"))
    LanguageCleaner.clean_file_parallel(input_file, output_file, language="chinese", reserve_newline=True)

