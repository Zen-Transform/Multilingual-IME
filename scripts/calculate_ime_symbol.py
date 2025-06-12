import os
import random
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def error_rate_to_string(error_rate: float) -> str:
    if error_rate == 0:
        return "0"
    return "r" + str(error_rate).replace(".", "-")


def get_random_line_efficient(filename):
    # The Con of this function is that the longer the line is the higher chance it gets selected
    with open(filename, "rb") as file:
        file_size = os.path.getsize(filename)
        random_position = random.randint(0, file_size - 1)
        file.seek(random_position)

        # Move backwards to find the start of the line
        while file.tell() > 0 and file.read(1) != b"\n":
            file.seek(file.tell() - 2)

        # Read the line
        return file.readline().decode("utf-8").strip()


DATA_AND_LABEL_SPLITTER = "\t"

if __name__ == "__main__":
    # Parameters
    total_samples = 1000
    error_rate = 0
    ime_types = ["japanese", "bopomofo", "cangjie", "pinyin", "english" ]
    # Settings
    SAVE_SYMBOL_FREQ = False
    SHOW_SYMBOL_FREQ_PLOT = True

    error_rate_str = error_rate_to_string(error_rate)
    BOPOMOFO_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_bopomofo_{error_rate_str}_test.txt"
    )
    CANGJIE_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_cangjie_{error_rate_str}_test.txt"
    )
    PINYIN_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_pinyin_{error_rate_str}_test.txt"
    )
    ENGLISH_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Test_Datasets\\labeled_wlen1_english_{error_rate_str}_test.txt"
    )
    JAPANESE_TOKEN_TEST_DATA_PATH = (
        f"Datasets\\Train_Datasets\\labeled_wlen1_japanese_{error_rate_str}_train.txt"
    )

    all_start_symbol_freq = {}
    all_end_symbol_freq = {}
    all_total_symbol_freq = {}

    for ime_type in ime_types:
        start_symbol_freq = {}
        end_symbol_freq = {}
        total_symbol_freq = {}

        with tqdm(total=total_samples, desc=f"Sampling {ime_type}") as pbar:
            while pbar.n < total_samples:

                if ime_type == "bopomofo":
                    sample_line = get_random_line_efficient(
                        BOPOMOFO_TOKEN_TEST_DATA_PATH
                    )
                elif ime_type == "cangjie":
                    sample_line = get_random_line_efficient(
                        CANGJIE_TOKEN_TEST_DATA_PATH
                    )
                elif ime_type == "pinyin":
                    sample_line = get_random_line_efficient(PINYIN_TOKEN_TEST_DATA_PATH)
                elif ime_type == "english":
                    sample_line = get_random_line_efficient(
                        ENGLISH_TOKEN_TEST_DATA_PATH
                    )
                elif ime_type == "japanese":
                    sample_line = get_random_line_efficient(
                        JAPANESE_TOKEN_TEST_DATA_PATH
                    )
                else:
                    raise ValueError(f"Unknown IME type: {ime_type}")
                # print("the line is", sample_line)
                splits = sample_line.split(DATA_AND_LABEL_SPLITTER)
                if len(splits) != 2:
                    continue

                sample_token, label = sample_line.split(DATA_AND_LABEL_SPLITTER)

                if label != "1":
                    continue

                start_symbol = sample_token[0]
                end_symbol = sample_token[-1]

                if start_symbol in start_symbol_freq:
                    start_symbol_freq[start_symbol] += 1
                else:
                    start_symbol_freq[start_symbol] = 1

                if end_symbol in end_symbol_freq:
                    end_symbol_freq[end_symbol] += 1
                else:
                    end_symbol_freq[end_symbol] = 1

                for symbol in sample_token:
                    if symbol in total_symbol_freq:
                        total_symbol_freq[symbol] += 1
                    else:
                        total_symbol_freq[symbol] = 1

                pbar.update(1)

        start_symbol_freq = sorted(
            start_symbol_freq.items(), key=lambda x: x[1], reverse=True
        )
        end_symbol_freq = sorted(
            end_symbol_freq.items(), key=lambda x: x[1], reverse=True
        )
        total_symbol_freq = sorted(
            total_symbol_freq.items(), key=lambda x: x[1], reverse=True
        )
        start_symbol_freq = [{x[0]: x[1]} for x in start_symbol_freq]
        end_symbol_freq = [{x[0]: x[1]} for x in end_symbol_freq]
        total_symbol_freq = [{x[0]: x[1]} for x in total_symbol_freq]

        print(f"Total Symbol Frequency for {ime_type}: {total_symbol_freq}")
        print(f"Start Symbol Frequency for {ime_type}: {start_symbol_freq}")
        print(f"End Symbol Frequency for {ime_type}: {end_symbol_freq}")
        print("\n")

        all_start_symbol_freq[ime_type] = start_symbol_freq
        all_end_symbol_freq[ime_type] = end_symbol_freq
        all_total_symbol_freq[ime_type] = total_symbol_freq

        if SAVE_SYMBOL_FREQ:
            # Save the results to a file
            with open(f"{ime_type}_symbol_freq.json", "w", encoding="utf-8") as f:

                json.dump(
                    {
                        "start_symbol_freq": start_symbol_freq,
                        "end_symbol_freq": end_symbol_freq,
                        "total_symbol_freq": total_symbol_freq,
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    def dict_to_dataframe(symbol_freqs):
        """
        Convert a dictionary of symbol frequencies to a DataFrame.

        Args:
            symbol_freqs (dict): Dictionary containing symbol frequencies for different IME types.

        Returns:
            pd.DataFrame: DataFrame containing the symbol frequencies.
        """
        all_symbols = {
            k for freq_list in symbol_freqs.values() for d in freq_list for k in d
        }
        df = (
            pd.DataFrame(index=symbol_freqs.keys(), columns=sorted(all_symbols))
            .fillna(0)
            .infer_objects(copy=False)
        )

        for ime_type, freqs in symbol_freqs.items():
            for freq in freqs:
                for symbol, count in freq.items():
                    df.at[ime_type, symbol] = count

        df = df.sort_index()

        return df

    from matplotlib.colors import LogNorm

    if SHOW_SYMBOL_FREQ_PLOT:
        start_symbol_df = dict_to_dataframe(all_start_symbol_freq)
        end_symbol_df = dict_to_dataframe(all_end_symbol_freq)
        total_symbol_df = dict_to_dataframe(all_total_symbol_freq)

        # Log scale normalization the df

        # Plotting the heatmaps
        fig_size = (12, 6)


        plt.figure(figsize=fig_size)
        sns.heatmap(
            start_symbol_df,
            cmap="Blues",
            cbar_kws={"label": "Frequency"},
            linewidths=0.05,
            linecolor="grey",
            norm=LogNorm(),
            xticklabels=True, 
            yticklabels=True
        )
        plt.title("IME Start Symbol Frequencies")
        plt.xlabel("Keystrokes")
        plt.ylabel("Input Methods")
        plt.show()

        # Plotting the heatmaps
        # Use log scale for better visibility
        plt.figure(figsize=fig_size)
        sns.heatmap(
            end_symbol_df,
            cmap="Blues",
            cbar_kws={"label": "Frequency"},
            linewidths=0.05,
            linecolor="grey",
            norm=LogNorm(),
            xticklabels=True, 
            yticklabels=True
        )
        plt.title("IME End Symbol Frequencies")
        plt.xlabel("Keystrokes")
        plt.ylabel("Input Methods")
        plt.show()
