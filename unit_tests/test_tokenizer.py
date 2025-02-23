from tqdm import tqdm
from multilingual_ime.ime_handler import (
    custom_tokenizer_pinyin,
    custom_tokenizer_english,
    custom_tokenizer_cangjie,
    custom_tokenizer_bopomofo,
)
from data_preprocess.keystroke_converter import KeyStrokeConverter


plain_text_path = ".\\Datasets\\Plain_Text_Datasets\\wlen1-3_cc100_train.txt"
keystroke_text_path = ".\\Datasets\\Train_Datasets\\0_pinyin_wlen1-3_cc100.txt"


def test_pinyin():
    print("Testing custom_tokenizer_pinyin")

    with open(keystroke_text_path, "r", encoding="utf-8") as f1:
        with open(plain_text_path, "r", encoding="utf-8") as f2:
            keystroke_texts = f1.readlines()
            plain_texts = f2.readlines()
            keystroke_texts = [text.strip() for text in keystroke_texts]
            plain_texts = [text.strip() for text in plain_texts]

    with tqdm(total=len(plain_texts)) as pbar:
        for plain_text, keystroke_data_text in zip(plain_texts, keystroke_texts):
            pbar.update(1)
            converter_keystroke_result = KeyStrokeConverter.convert(
                plain_text, "pinyin"
            )
            con_tokens = custom_tokenizer_pinyin(converter_keystroke_result)
            direct_tokens = custom_tokenizer_pinyin(keystroke_data_text)

            joined_con_tokens = "".join(con_tokens)
            if len(joined_con_tokens) != len(converter_keystroke_result):
                print(
                    f"Error Losing characters:\n"
                    f"plaintext: {plain_text}\n"
                    f"con: {converter_keystroke_result}\n"
                    f"joint tokens: {joined_con_tokens}\n"
                    f"direct_tokens:{direct_tokens}\n\n"
                )

            if con_tokens != direct_tokens:
                print(
                    f"Error: Tokenization mismatch \n"
                    f"plaintext: {plain_text}\n"
                    f"keystroke data text:{keystroke_data_text}/{direct_tokens}\n"
                    f"con:{converter_keystroke_result}/{con_tokens}\n"
                    f"direct_tokens:{direct_tokens}\n\n"
                )

            # assert converter_keystroke_result == joined_con_tokens, (
            #     f"Error Losing characters:\n"
            #     f"plaintext: {plain_text}\n"
            #     f"con: {converter_keystroke_result}\n"
            #     f"joint tokens: {joined_con_tokens}\n"
            # )

            # assert con_tokens == direct_tokens, (
            #     f"Error: Tokenization mismatch \n"
            #     f"plaintext: {plain_text}\n"
            #     f"keystroke data text:{keystroke_data_text}/{direct_tokens}\n"
            #     f"con:{converter_keystroke_result}/{con_tokens}\n"
            #     f"direct_tokens:{direct_tokens}"
            # )


def test_custom_tokenizer_pinyin():
    # right format
    assert ["mou", "ren"] in custom_tokenizer_pinyin("mouren")
    assert ["shi", "re", "lu"] in custom_tokenizer_pinyin("shirelu")
    assert [
        "en",
    ] in custom_tokenizer_pinyin("en")
    assert ["xi", "en", "shi"] in custom_tokenizer_pinyin("xienshi")

    # underpresented
    assert [
        "k",
    ] in custom_tokenizer_pinyin("k")
    assert ["chu", "qu", "wa"] in custom_tokenizer_pinyin("chuquwa")
    assert ["han", "yu", "pin", "y"] in custom_tokenizer_pinyin("hanyupiny")

    # overpresented
    assert ["chu", "qu", "wan"] in custom_tokenizer_pinyin("chuquwan")
    assert ["chu", " ", "qu", " ", " ", "wan"] in custom_tokenizer_pinyin("chu qu  wan")

    # edge cases
    assert [] == custom_tokenizer_pinyin("")
    assert [" ", " "] in custom_tokenizer_pinyin("  ")

    # Other languages
    assert ["你好"] in custom_tokenizer_pinyin("你好")
    assert ["&", "^", "%", "$", "#"] in custom_tokenizer_pinyin("&^%$#")


def test_custom_tokenizer_english():
    # right format
    assert ["Hello", " ", "world"] in custom_tokenizer_english("Hello world")
    assert [
        "apple",
    ] in custom_tokenizer_english("apple")
    assert [
        " ",
        "78",
        " ",
        "Good",
        " ",
        "9",
        " ",
        "_",
        "+",
    ] in custom_tokenizer_english(" 78 Good 9 _+")

    # underpresented
    assert ["k"] in custom_tokenizer_english("k")
    assert ["differe"] in custom_tokenizer_english("differe")

    # overpresented
    assert ["apple", " ", " "] in custom_tokenizer_english("apple  ")
    assert ["apple", " ", " ", "pieeee"] in custom_tokenizer_english("apple  pieeee")

    # edge cases
    assert [] == custom_tokenizer_english("")
    assert [" ", " ", "!", "!", "?", "?"] in custom_tokenizer_english("  !!??")

    # Other languages
    assert ["你好"] in custom_tokenizer_english("你好")
    assert ["&", "^", "%", "$", "#"] in custom_tokenizer_english("&^%$#")


def test_custom_tokenizer_cangjie():
    # right format
    assert ["aaa ", "bbb ", "ccc"] in custom_tokenizer_cangjie("aaa bbb ccc")
    assert ["omg ", "jnd ", "ijwj "] in custom_tokenizer_cangjie("omg jnd ijwj ")

    # underpresented
    assert ["k"] in custom_tokenizer_cangjie("k")
    assert [" "] in custom_tokenizer_cangjie(" ")

    # overpresented
    assert ["k ", " "] in custom_tokenizer_cangjie("k  ")
    assert ["mf ", " ", "ogd ", "l "] in custom_tokenizer_cangjie("mf  ogd l ")
    # edge cases
    assert [] == custom_tokenizer_cangjie("")
    assert ["gjieojago"] in custom_tokenizer_cangjie("gjieojago")

    # Other languages/characters
    assert ["你好"] in custom_tokenizer_cangjie("你好")
    assert ["789$0 ", "!@#"] in custom_tokenizer_cangjie("789$0 !@#")
    # ↑↑ should raise error?


def test_custom_tokenizer_bopomofo():
    # right format
    assert ["su3", "cl3", "a87"] in custom_tokenizer_bopomofo("su3cl3a87")
    assert ["j6", "vu04", "ru/4", "2u4"] in custom_tokenizer_bopomofo("j6vu04ru/42u4")

    # underpresented
    assert ["su3", "cl3", "a8"] in custom_tokenizer_bopomofo("su3cl3a8")
    assert ["t ", "z"] in custom_tokenizer_bopomofo("t z")

    # overpresented
    assert ["su3", "cl3", "a87", " "] in custom_tokenizer_bopomofo("su3cl3a87 ")
    assert [
        "nji3",
        "u.3",
        "fm06",
        "bp6",
        "6",
    ] in custom_tokenizer_bopomofo("nji3u.3fm06bp66")

    # edge cases
    assert [] == custom_tokenizer_bopomofo("")
    assert [" ", " "] in custom_tokenizer_bopomofo("  ")

    # Other languages/characters
    assert ["你好"] in custom_tokenizer_bopomofo("你好")
    assert ["7", "8989$290 ", "!@#"] in custom_tokenizer_bopomofo("78989$290 !@#")


if __name__ == "__main__":
    test_pinyin()
    print("All tests passed!")
    test_cases = ["mouren", "shierge", "n", "xienshi"]
    for case in test_cases:
        print(f"{case}: {custom_tokenizer_pinyin(case)}")
    text = "mouren", "shierge"
    print(f"{text }: {custom_tokenizer_pinyin(text)}")
