
import pandas as pd
from colorama import Fore, Style

if __name__ == "__main__":
    file_path = r""
    df = pd.read_csv(file_path, encoding="utf-8")
    print(df.columns.to_list())

    df["correct"] = df["correct"].apply(lambda x: 1 if x is True else 0)
    total_correct = df["correct"].sum()
    total_test = len(df)

    # Show report
    print(Fore.GREEN + "-" * 50 + Style.RESET_ALL)
    print(Fore.YELLOW + "Report:" + Style.RESET_ALL)
    print(Fore.CYAN + file_path.rsplit("\\", maxsplit=1)[-1] + Style.RESET_ALL)
    print(Fore.MAGENTA + "Total test case: " + Style.RESET_ALL + f"{total_test}")
    print(Fore.MAGENTA + "Total correct: " + Style.RESET_ALL + f"{total_correct}")
    print(
        Fore.BLUE
        + "Accuracy (correct/total_test_case): "
        + Style.RESET_ALL
        + f"{total_correct / total_test}"
    )
    print(Fore.BLUE + "Avg BLEUn4: " + Style.RESET_ALL + f"{df['BLEUn4'].mean():.3f}")
    print(Fore.BLUE + "Avg BLEUn3: " + Style.RESET_ALL + f"{df['BLEUn3'].mean():.3f}")
    print(Fore.BLUE + "Avg BLEUn2: " + Style.RESET_ALL + f"{df['BLEUn2'].mean():.3f}")
    print(Fore.BLUE + "Avg BLEUn1: " + Style.RESET_ALL + f"{df['BLEUn1'].mean():.3f}")
    print(Fore.BLUE + "Avg unigram: " + Style.RESET_ALL + f"{df['unigram'].mean():.3f}")
    print(Fore.BLUE + "Avg bigrams: " + Style.RESET_ALL + f"{df['bigrams'].mean():.3f}")
    print(Fore.BLUE + "Avg trigrams: " + Style.RESET_ALL + f"{df['trigrams'].mean():.3f}")
    print(Fore.BLUE + "Avg fourgrams: " + Style.RESET_ALL + f"{df['fourgrams'].mean():.3f}")
    print(
        Fore.RED + "Avg time spend: " + Style.RESET_ALL + f"{df['time_spend'].mean():.3f}"
    )
    print(Fore.GREEN + "-" * 50 + Style.RESET_ALL)
