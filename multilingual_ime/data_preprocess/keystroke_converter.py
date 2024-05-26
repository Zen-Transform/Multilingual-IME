import os

from pypinyin import pinyin, lazy_pinyin, Style
import multiprocessing
from tqdm import tqdm

class KeyStrokeConverter:

    @classmethod
    def convert(cls, input_string:str, convert_type:str) -> str:
        """
        Convert the input string to the specified type of keystroke

        Args:
            input_string (str): The input string
            convert_type (str): The type of keystroke to convert to. 
                                Valid values are "english", "cangjie", "bopomofo", "pinyin"
        
        Raises: 
            ValueError: If convert_type is not valid
        
        Returns:
            str: The converted string
        """


        if convert_type == "english":
            converted_string = cls._StringToEnglishKey(input_string)
        elif convert_type == "cangjie":
            converted_string = cls._StringToCangjieKey(input_string)
        elif convert_type == "bopomofo":
            converted_string = cls._StringToBopomofoKey(input_string)
        elif convert_type == "pinyin":
            converted_string = cls._StringToPinyinKey(input_string)
        else:
            raise ValueError("Invalid convert_type: " + convert_type)

        return converted_string
    

    @classmethod
    def _StringToCangjieKey(cls, input_string:str) -> str:

        def cangjie_key_map_create() -> dict:  # Deprecated
            files = ["Cangjie5.txt", "Cangjie-markers.txt"]

            cangjie_key_map_dict = {}
            counter = 0
            for filename in files:
                with open(os.path.dirname(__file__) + "\\" + filename, "r", encoding="utf-8") as file:
                    lines = file.readlines()[:118190]  # beacuse the remain lines are symbols
                    
                    for line in lines:
                        parts = line.strip().split()
                        word, cangjie_key = parts[0], parts[1]
                        if cangjie_key_map_dict.get(word) is not None:
                            counter += 1
                            cangjie_key_map_dict[word] = cangjie_key_map_dict[word] + [cangjie_key]

                        cangjie_key_map_dict.setdefault(word, [cangjie_key])
            print("Duplicate words: " + str(counter))

            with open(os.path.dirname(__file__) + "\\cangjie_key_map.txt", "w", encoding="utf-8") as file:
                for key, value in cangjie_key_map_dict.items():
                    file.write(key + " " + " ".join(value) + "\n")
            return cangjie_key_map_dict

        def setup_cangjie_key_map():  # todo: make the dict not load every time
            file = "cangjie_key_map.txt"
            cangjie_key_map_dict = {}
            with open(os.path.dirname(__file__) + "\\" + file , "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    word, cangjie_keys = parts[0], parts[1:]
                    cangjie_key = cangjie_keys[0]
                    cangjie_key_map_dict[word] = cangjie_key + " "
            return cangjie_key_map_dict

        cangjie_map_dict = setup_cangjie_key_map()
        
        result = ""
        for line in input_string:
            have_newline = line.find("\n") != -1
            line = line.strip()
            result += ''.join([cangjie_map_dict.get(char, char) for char in line]) + ("\n" if have_newline else "")

        return result


    @classmethod
    def _StringToBopomofoKey(cls, input_string: str) -> str:

        def bopomofo_to_keystroke(bopomofo: str) -> str:
            map = {
                "ㄅ": "1", "ㄆ": "q", "ㄇ": "a",
                "ㄈ": "z", "ㄉ": "2", "ㄊ": "w",
                "ㄋ": "s", "ㄌ": "x", "ㄍ": "e",
                "ㄎ": "d", "ㄏ": "c", "ㄐ": "r",
                "ㄑ": "f", "ㄒ": "v", "ㄓ": "5",
                "ㄔ": "t", "ㄕ": "g", "ㄖ": "b",
                "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
                "ㄚ": "8", "ㄛ": "i", "ㄜ": "k",
                "ㄝ": ",", "ㄞ": "9", "ㄟ": "o",
                "ㄠ": "l", "ㄡ": ".", "ㄢ": "0",
                "ㄣ": "p", "ㄤ": ";", "ㄥ": "/",
                "ㄦ": "-", "ㄧ": "u", "ㄨ": "j",
                "ㄩ": "m", "ˊ": "6", "ˇ": "3",
                "ˋ": "4",  "˙": "7",
            }

            keystroke = ""
            try:
                if bopomofo[0] in cls.full_width_map.keys():  # if the word is full width speical character
                    for word in bopomofo:
                        keystroke += cls.full_width_map.get(word, word)
                else:
                        
                    for word in bopomofo:
                        keystroke += map.get(word, word)
                    
                    if not keystroke.endswith(("6", "3", "4", "7")):  # if the word is tone 1 add a space
                        keystroke += " "
                
            except KeyError:
                    print("Invalid bopomofo: " + bopomofo)

                    # raise ValueError("Invalid bopomofo: " + bopomofo)
            
            return keystroke
        result = ""
        for line in input_string:
            have_newline = line.find("\n") != -1
            line = line.strip()
            BOPOMOFO_result = [pin[0] for pin in pinyin(line, style=Style.BOPOMOFO)]
            keystorke_result = [bopomofo_to_keystroke(word) for word in BOPOMOFO_result]
            keystorke_result = "".join(keystorke_result)
            result += keystorke_result + ("\n" if have_newline else "")

        return result


    @classmethod
    def _StringToPinyinKey(cls, input_string:str) -> str:
        PINYIN_result = [pin[0] for pin in pinyin(input_string, style=Style.NORMAL)]


        # result = [pin if pin[0] in cls.full_width_map.keys() else pin for pin in PINYIN_result]
        keystroke = ""
        for pin in PINYIN_result:
            if pin[0] in cls.full_width_map.keys():
                for word in pin:
                    keystroke += cls.full_width_map.get(word, word)
            else:
                keystroke += pin

        return keystroke


    @classmethod
    def _StringToEnglishKey(cls, input_string: str) -> str:
        return input_string

    CTRL_KEY = "®"
    full_width_map = {
        "１": "1", "２": "2", "３": "3",
        "４": "4", "５": "5", "６": "6",
        "７": "7", "８": "8", "９": "9",
        "０": "0", 
        
        "，": CTRL_KEY + ",", "。": CTRL_KEY + ".",
        "；": CTRL_KEY + ";", "：": CTRL_KEY + ":",
        "、": CTRL_KEY + "'",
        "？": CTRL_KEY + "?", "！": CTRL_KEY + "!",
        "（": CTRL_KEY + "(", "）": CTRL_KEY + ")",  # fixme: not sure if this is correct 
        "【": CTRL_KEY + "[", "】": CTRL_KEY + "]",
        "｛": CTRL_KEY + "{", "｝": CTRL_KEY + "}",

        "「": CTRL_KEY + "[", "」": CTRL_KEY + "]",  # fixme: not sure if this is correct
    }

    @classmethod
    def _full_width_to_half_width(cls, input_char: str) -> str:
            return cls.full_width_map.get(input_char, input_char)


    @staticmethod
    def _convert_file_chunk(input_queue, output_queue, convert_type:str):
        while True:
            chunk_index, chunk = input_queue.get()
            if chunk is None:
                output_queue.put(None)  # Signal the main process that this worker is done
                break
            cleaned_chunk = KeyStrokeConverter.convert(chunk, convert_type=convert_type)
            output_queue.put((chunk_index, cleaned_chunk))


    @staticmethod
    def convert_file_parallel(input_file_path:str, output_file_path:str, convert_type:str, num_processes:int=4): # fix: make it cleaner
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
            p = multiprocessing.Process(target=KeyStrokeConverter._convert_file_chunk, args=(input_queue, output_queue, convert_type))
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
        print(f"Conversion Success: {output_file_path}")

if __name__ == '__main__':
    input_string = "僅頒行政院長陳建仁今\n（16）日出席\n「112年鳳凰獎楷模表揚典禮」，頒獎表揚74名獲獎義消"
    convert_type = "cangjie"
    print(input_string)
    print(KeyStrokeConverter.convert(input_string, convert_type))
    # input_file =  "..\\Plain_Text_Datasets\\bbb.txt"
    # output_file =  "..\\Key_Stroke_Datasets\\aaa.txt"
    # KeyStrokeConverter.convert_file_parallel(input_file, output_file, convert_type="cangjie", num_processes=4)