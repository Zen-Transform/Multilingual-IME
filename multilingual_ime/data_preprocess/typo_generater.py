import random

from tqdm import tqdm
import multiprocessing
import string

characters = string.digits + string.ascii_letters + ",.;/-'=[]"
KEYSTROKES = [c for c in characters]


def character_mapping(input: str) -> list[str]:
    if input == "1":
        return ["2", "q", "w"]
    elif input == "2":
        return ["1", "3", "q", "w", "e"]
    elif input == "3":
        return ["2", "4", "w", "e", "r"]
    elif input == "4":
        return ["3", "5", "e", "r", "t"]
    elif input == "5":
        return ["4", "6", "r", "t", "y"]
    elif input == "6":
        return ["5", "7", "t", "y", "u"]
    elif input == "7":
        return ["6", "8", "y", "u", "i"]
    elif input == "8":
        return ["7", "9", "u", "i", "o"]
    elif input == "9":
        return ["8", "0", "i", "o", "p"]
    elif input == "0":
        return ["9", "-", "o", "p", "["]
    elif input == "-":
        return ["0", "=", "p", "[", "]"]
    elif input == "=":
        return ["-", "\\", "[", "]"]
    elif input == "q":
        return ["1", "2", "w", "a", "s"]
    elif input == "Q":
        return ["1", "2", "W", "A", "S"]
    elif input == "w":
        return ["1", "2", "3", "q", "a", "s", "d", "e"]
    elif input == "W":
        return ["1", "2", "3", "E", "A", "S", "D", "Q"]
    elif input == "e":
        return ["2", "3", "4", "w", "s", "d", "f", "r"]
    elif input == "E":
        return ["2", "3", "4", "W", "S", "D", "F", "R"]
    elif input == "r":
        return ["3", "4", "5", "e", "d", "f", "g", "t"]
    elif input == "R":
        return ["3", "4", "5", "E", "D", "F", "G", "T"]
    elif input == "t":
        return ["4", "5", "6", "r", "f", "g", "h", "y"]
    elif input == "T":
        return ["4", "5", "6", "R", "F", "G", "H", "Y"]
    elif input == "y":
        return ["5", "6", "7", "t", "g", "h", "j", "u"]
    elif input == "Y":
        return ["5", "6", "7", "T", "G", "H", "J", "U"]
    elif input == "u":
        return ["6", "7", "8", "y", "h", "j", "k", "i"]
    elif input == "U":
        return ["6", "7", "8", "Y", "H", "J", "K", "I"]
    elif input == "i":
        return ["7", "8", "9", "u", "j", "k", "l", "o"]
    elif input == "I":
        return ["7", "8", "9", "U", "J", "K", "L", "O"]
    elif input == "o":
        return ["8", "9", "0", "i", "k", "l", ";" "p"]
    elif input == "O":
        return ["8", "9", "0", "I", "K", "L", ";" "P"]
    elif input == "p":
        return ["9", "0", "-", "o", "l", ";", "'", "["]
    elif input == "P":
        return ["9", "0", "-", "O", "L", ";", "'", "["]
    elif input == "[":
        return ["0", "-", "=", "p", ";", "'", "]"]
    elif input == "]":
        return ["-", "=", "[", "'", "\\"]
    elif input == "a":
        return ["q", "w", "s", "z", "x"]
    elif input == "A":
        return ["Q", "W", "S", "Z", "X"]
    elif input == "s":
        return ["q", "w", "e", "a", "d", "z", "x", "c"]
    elif input == "S":
        return ["Q", "W", "E", "A", "D", "Z", "X", "C"]
    elif input == "d":
        return ["w", "e", "r", "s", "f", "x", "c", "v"]
    elif input == "D":
        return ["W", "E", "R", "S", "F", "X", "C", "V"]
    elif input == "f":
        return ["e", "r", "t", "d", "g", "c", "v", "b"]
    elif input == "F":
        return ["E", "R", "T", "D", "G", "C", "V", "B"]
    elif input == "g":
        return ["r", "t", "y", "f", "h", "v", "b", "n"]
    elif input == "G":
        return ["R", "T", "Y", "F", "H", "V", "B", "N"]
    elif input == "h":
        return ["t", "y", "u", "g", "j", "b", "n", "m"]
    elif input == "H":
        return ["T", "Y", "U", "G", "J", "B", "N", "M"]
    elif input == "j":
        return ["y", "u", "i", "h", "k", "n", "m", ","]
    elif input == "J":
        return ["Y", "U", "I", "H", "K", "N", "M", ","]
    elif input == "k":
        return ["u", "i", "o", "j", "l", "m", ",", "."]
    elif input == "K":
        return ["U", "I", "O", "J", "L", "M", ",", "."]
    elif input == "l":
        return ["i", "o", "p", "k", ";", ".", "/"]
    elif input == "L":
        return ["I", "O", "P", "K", ";", ".", "/"]
    elif input == ";":
        return ["o", "p", "[", "l", "'", ".", "/"]
    elif input == "'":
        return ["p", "[", "]", ";", "/"]
    elif input == "z":
        return ["a", "s", "x"]
    elif input == "Z":
        return ["A", "S", "X"]
    elif input == "x":
        return ["a", "s", "d", "z", "c"]
    elif input == "X":
        return ["A", "S", "D", "Z", "C"]
    elif input == "c":
        return ["s", "d", "f", "x", "v"]
    elif input == "C":
        return ["S", "D", "F", "X", "V"]
    elif input == "v":
        return ["d", "f", "g", "c", "b"]
    elif input == "V":
        return ["D", "F", "G", "C", "B"]
    elif input == "b":
        return ["f", "g", "h", "v", "n"]
    elif input == "B":
        return ["F", "G", "H", "V", "N"]
    elif input == "n":
        return ["g", "h", "j", "b", "m"]
    elif input == "N":
        return ["G", "H", "J", "B", "M"]
    elif input == "m":
        return ["h", "j", "k", "n", ","]
    elif input == "M":
        return ["H", "J", "K", "N", ","]
    elif input == ",":
        return ["j", "k", "l", "m", "."]
    elif input == ".":
        return ["k", "l", ";", ",", "/"]
    elif input == "/":
        return ["l", ";", "'", "."]
    else:
        return KEYSTROKES


class TypoGenerater:

    @staticmethod
    def _generate_8adjacency(input_string: str, error_rate: float) -> str:

        if not (0 <= error_rate <= 1):
            raise ValueError("error_rate should be between 0 and 1")

        typo_index = []
        for i in range(len(input_string)):
            try:
                if random.random() < error_rate:
                    typo_index.append(i)
            except IndexError:
                print("IndexError: ", i, input_string)

        for i in typo_index:
            if i >= len(input_string):
                i = random.randrange(0, len(input_string))
            e = random.randrange(0, 4)
            try:
                if e == 0:  # Swap
                    if i < len(input_string) - 1:
                        input_string = (
                            input_string[:i]
                            + input_string[i + 1]
                            + input_string[i]
                            + input_string[i + 2 :]
                        )
                elif e == 1:  # Delete
                    if i < len(input_string) - 1:
                        input_string = input_string[:i] + input_string[i + 1 :]
                elif e == 2:  # Add
                    input_string = (
                        input_string[:i] + random.choice(KEYSTROKES) + input_string[i:]
                    )
                elif e == 3:  # Replace
                    input_string = (
                        input_string[:i]
                        + random.choice(character_mapping(input_string[i]))
                        + input_string[i + 1 :]
                    )
                else:
                    raise ValueError(
                        "error rate should be between 0 and 3, Should not reach here"
                    )
            except IndexError:
                print("IndexError: ", i, input_string)

        return input_string

    @staticmethod
    def generate(input_string: str, error_type: str, error_rate: float) -> str:
        """
        Generate typos for the input string

        Args:
            input_string (str): The input string
            error_type (str): The type of error, "random" or "8adjacency"
            error_rate (float): The error rate (0 <= error_rate <= 1)

        Raises:
            ValueError: If error_rate is not between 0 and 1
        Returns:
            str: The string with typos
        """

        if not (0 <= error_rate <= 1):
            raise ValueError("error_rate should be between 0 and 1")

        if error_rate == 0:
            return input_string

        if error_type == "random":
            lines = input_string.split("\n")
            for i in range(len(lines)):
                lines[i] = TypoGenerater._generate_random(lines[i], error_rate)
            return "\n".join(lines)
        elif error_type == "8adjacency":
            lines = input_string.split("\n")
            for i in range(len(lines)):
                lines[i] = TypoGenerater._generate_8adjacency(lines[i], error_rate)
            return "\n".join(lines)
        else:
            raise ValueError(
                "Error: error_type '{}' is not supported".format(error_type)
            )

    def _generate_random(input_string: str, error_rate: float) -> str:
        """
        Generate typos using random error for the input string

        Args:
            input_string (str): The input string
            error_rate (float): The error rate (0 <= error_rate <= 1)
        Returns:
            str: The string with typos generated using random error
        """

        typo_index = []
        for i in range(len(input_string)):
            if random.random() < error_rate:
                typo_index.append(i)

        for i in typo_index:
            e = random.randrange(0, 4)
            if e == 0:  # Swap
                if i < len(input_string) - 1:
                    input_string = (
                        input_string[:i]
                        + input_string[i + 1]
                        + input_string[i]
                        + input_string[i + 2 :]
                    )
            elif e == 1:  # Delete
                if i < len(input_string) - 1:
                    input_string = input_string[:i] + input_string[i + 1 :]
            elif e == 2:  # Add
                input_string = (
                    input_string[:i] + random.choice(KEYSTROKES) + input_string[i:]
                )
            elif e == 3:  # Replace
                input_string = (
                    input_string[:i] + random.choice(KEYSTROKES) + input_string[i + 1 :]
                )
            else:
                raise ValueError("e should be between 0 and 3, Should not reach here")

        return input_string

    @staticmethod
    def _generate_error_chunk(
        input_queue, output_queue, error_type: str, error_rate: float
    ):
        while True:
            chunk_index, chunk = input_queue.get()
            if chunk is None:
                output_queue.put(
                    None
                )  # Signal the main process that this worker is done
                break
            cleaned_chunk = TypoGenerater.generate(
                chunk, error_type=error_type, error_rate=error_rate
            )
            output_queue.put((chunk_index, cleaned_chunk))

    @staticmethod
    def generate_file_parallel(
        input_file_path: str,
        output_file_path: str,
        error_type: str,
        error_rate: float,
        num_processes: int = 4,
    ):  # fix: make it cleaner
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
        with open(input_file_path, "r", encoding="utf-8") as f:
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
            p = multiprocessing.Process(
                target=TypoGenerater._generate_error_chunk,
                args=(input_queue, output_queue, error_type, error_rate),
            )
            processes.append(p)
            p.start()

        # Collect and reorder cleaned chunks
        cleaned_chunks = [None] * chunk_index
        workers_done = 0
        with tqdm(total=chunk_index) as pbar:
            while workers_done < num_processes:
                try:
                    chunk_info = output_queue.get(
                        timeout=1
                    )  # Timeout to prevent hanging
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
        with open(output_file_path, "w", encoding="utf-8") as f:
            for cleaned_chunk in cleaned_chunks:
                if cleaned_chunk is not None:
                    f.write(cleaned_chunk)
        print(f"Typo Generation Success: {output_file_path}")


if __name__ == "__main__":
    # input_string = "hello world"
    # count = 0
    # SAMPLE_SIZE = 10000
    # for i in range(SAMPLE_SIZE):
    #     reuslt = TypoGenerater.generate(input_string, error_type="random", error_rate=0.001)
    #     if reuslt != input_string:
    #         count += 1
    # print("Error rate: {}/{} : {}".format(count, SAMPLE_SIZE, count / SAMPLE_SIZE))

    ERROR_RATE = 0.1
    ERROR_TYPE = "random"

    input_file = "..\\Key_Stroke_Datasets\\aaa.txt"
    print(input_file)
    output_file = "..\\Key_Stroke_Datasets\\bbb.txt"
    TypoGenerater.generate_file_parallel(
        input_file, output_file, ERROR_TYPE, ERROR_RATE
    )
