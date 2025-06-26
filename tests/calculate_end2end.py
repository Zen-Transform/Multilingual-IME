import os
from pathlib import Path
import pandas as pd
from colorama import Fore, Style

def parse_name(name:str) -> tuple[str, str, str]:
    parts = name.split("_")
    print(parts)
    type_part = parts[3]
    n_token_part = parts[4].replace("-tokens", "")
    error_rate = parts[5].replace("r","").replace("-", ".")
    return type_part, n_token_part, error_rate


if __name__ == "__main__":
    dir_path = Path(__file__).parent.parent / "reports"

    all_df = pd.DataFrame()

    for file_name in os.listdir(dir_path):
        if not file_name.endswith(".csv"):
            continue
        df = pd.read_csv(dir_path / file_name, encoding="utf-8")

        df["e2e_correct"] = df["e2e_correct"].apply(lambda x: 1 if x is True else 0)
        df["sep_correct"] = df["sep_correct"].apply(lambda x: 1 if x is True else 0)
        total_e2e_correct = df["e2e_correct"].sum()
        total_sep_correct = df["sep_correct"].sum()
        total_test = len(df)

        mix_type, n_token, error_rate = parse_name(file_name)


        # Show report
        print(Fore.GREEN + "-" * 50 + Style.RESET_ALL)
        print(Fore.YELLOW + "Report:" + Style.RESET_ALL)
        print(Fore.CYAN + "{} {} {}:".format(mix_type, n_token, error_rate) + Style.RESET_ALL)
        print(Fore.MAGENTA + "Total test case: " + Style.RESET_ALL + f"{total_test}")
        print(
            Fore.MAGENTA + "Total correct: " + Style.RESET_ALL + f"{total_e2e_correct}"
        )
        print(
            Fore.MAGENTA
            + "Total sep correct: "
            + Style.RESET_ALL
            + f"{total_sep_correct}"
        )
        print(
            Fore.BLUE
            + "Accuracy (correct/total_test_case): "
            + Style.RESET_ALL
            + f"{total_e2e_correct / total_test}"
        )
        print(
            Fore.BLUE
            + "Sep Accuracy (sep_correct/total_test_case): "
            + Style.RESET_ALL
            + f"{total_sep_correct / total_test}"
        )
        print(
            Fore.BLUE + "Avg BLEUn4: " + Style.RESET_ALL + f"{df['BLEUn4'].mean():.3f}"
        )
        print(
            Fore.BLUE + "Avg BLEUn3: " + Style.RESET_ALL + f"{df['BLEUn3'].mean():.3f}"
        )
        print(
            Fore.BLUE + "Avg BLEUn2: " + Style.RESET_ALL + f"{df['BLEUn2'].mean():.3f}"
        )
        print(
            Fore.BLUE + "Avg BLEUn1: " + Style.RESET_ALL + f"{df['BLEUn1'].mean():.3f}"
        )
        print(
            Fore.BLUE
            + "Avg unigram: "
            + Style.RESET_ALL
            + f"{df['unigram'].mean():.3f}"
        )
        print(
            Fore.BLUE
            + "Avg bigrams: "
            + Style.RESET_ALL
            + f"{df['bigrams'].mean():.3f}"
        )
        print(
            Fore.BLUE
            + "Avg trigrams: "
            + Style.RESET_ALL
            + f"{df['trigrams'].mean():.3f}"
        )
        print(
            Fore.BLUE
            + "Avg fourgrams: "
            + Style.RESET_ALL
            + f"{df['fourgrams'].mean():.3f}"
        )
        print(
            Fore.RED
            + "Avg time spend: "
            + Style.RESET_ALL
            + f"{df['time_spend'].mean():.3f}"
        )
        print(Fore.GREEN + "-" * 50 + Style.RESET_ALL)

        temp_df = pd.DataFrame([{
            "mix_type": mix_type,
            "n_token": n_token,
            "error_rate": error_rate,
            "total_e2e_acc": total_e2e_correct/ total_test,
            "total_sep_acc": total_sep_correct/ total_test,
            "BLEUn4": df["BLEUn4"].mean(),
            "time_spend": df["time_spend"].mean()
        }])
        all_df = pd.concat([all_df, temp_df], ignore_index=True)
    
    all_df.to_csv("system_test_all.csv",index=False, encoding="utf-8")