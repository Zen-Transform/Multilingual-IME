import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


def prepare_data_for_heatmap(symbol_freqs):
    """
    Prepare data for heatmap visualization from symbol frequencies.

    Args:
        symbol_freqs (dict): Dictionary containing symbol frequencies for different IME types.

    Returns:
        pd.DataFrame: DataFrame containing the start, end, and total symbol frequencies.
    """
    start_freqs = {k: v["start_symbol_freq"] for k, v in symbol_freqs.items()}
    end_freqs = {k: v["end_symbol_freq"] for k, v in symbol_freqs.items()}
    total_freqs = {k: v["total_symbol_freq"] for k, v in symbol_freqs.items()}

    start_total_symbols = set(start_freqs.keys()).union(*start_freqs.values())
    end_total_symbols = set(end_freqs.keys()).union(*end_freqs.values())
    total_total_symbols = set(total_freqs.keys()).union(*total_freqs.values())

    start_df = pd.DataFrame(
        index=start_total_symbols, columns=start_freqs.keys()
    ).fillna(0)
    end_df = pd.DataFrame(index=end_total_symbols, columns=end_freqs.keys()).fillna(0)
    total_df = pd.DataFrame(
        index=total_total_symbols, columns=total_freqs.keys()
    ).fillna(0)

    return start_df, end_df, total_df


if __name__ == "__main__":
    json_files = [
        "bopomofo_symbol_freq.json",
        "cangjie_symbol_freq.json",
        "pinyin_symbol_freq.json",
        "english_symbol_freq.json",
    ]
    ime_types = ["bopomofo", "cangjie", "pinyin", "english"]

    ime_symbol_freqs = {}

    for ime_type, json_file in zip(ime_types, json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            ime_symbol_freqs[ime_type] = json.load(f)

    # Convert the dictionary to a DataFrame for easier plotting
    start_df, end_df, total_df = prepare_data_for_heatmap(ime_symbol_freqs)

    print(start_df.head())
    # Plot the heatmap for total symbols

    # # Draw the heatmap for start symbols
    # plt.figure(figsize=(10, 8))
    # sns.heatmap()
