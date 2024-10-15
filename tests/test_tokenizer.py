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


def test_custom_tokenizer_pinyin():
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
    assert custom_tokenizer_pinyin("mouren") == ["mou", "ren"]
    assert custom_tokenizer_pinyin("shirelu") == ["shi", "re", "lu"]
    assert custom_tokenizer_pinyin("n") == ["n"]
    assert custom_tokenizer_pinyin("xienshi") == ["xien", "shi"]

    # underpresented
    assert custom_tokenizer_pinyin("k") == ["k"]
    assert custom_tokenizer_pinyin("chuquwa") == ["chu", "qu", "wa"]
    assert custom_tokenizer_pinyin("hanyupiny") == ["han", "yu", "pin", "y"]

    # overpresented
    assert custom_tokenizer_pinyin("chuquwan") == ["chu", "qu", "wan"]
    assert custom_tokenizer_pinyin("chu qu  wan") == ["chu", " ", "qu", " ", " ", "wan"]

    # edge cases
    assert custom_tokenizer_pinyin("") == []
    assert custom_tokenizer_pinyin("  ") == [" ", " "]

    # Other languages
    assert custom_tokenizer_pinyin("你好") == ["你好"]
    assert custom_tokenizer_pinyin("&^%$#") == ["&", "^", "%", "$", "#"]


def test_custom_tokenizer_english():
    # right format
    assert custom_tokenizer_english("Hello world") == ["Hello", " ", "world"]
    assert custom_tokenizer_english("apple") == ["apple"]
    assert custom_tokenizer_english(" 78 Good 9 _+") == [
        " ",
        "78",
        " ",
        "Good",
        " ",
        "9",
        " ",
        "_",
        "+",
    ]
    # underpresented
    assert custom_tokenizer_english("k") == ["k"]
    assert custom_tokenizer_english("differe") == ["differe"]

    # overpresented
    assert custom_tokenizer_english("apple  ") == ["apple", " ", " "]
    assert custom_tokenizer_english("apple  pieeee") == ["apple", " ", " ", "pieeee"]

    # edge cases
    assert custom_tokenizer_english("") == []
    assert custom_tokenizer_english("  !!??") == [" ", " ", "!", "!", "?", "?"]

    # Other languages
    assert custom_tokenizer_english("你好") == ["你好"]
    assert custom_tokenizer_english("&^%$#") == ["&", "^", "%", "$", "#"]


def test_custom_tokenizer_cangjie():
    # right format
    assert custom_tokenizer_cangjie("aaa bbb ccc") == ["aaa ", "bbb ", "ccc"]
    assert custom_tokenizer_cangjie("omg jnd ijwj ") == ["omg ", "jnd ", "ijwj "]

    # underpresented
    assert custom_tokenizer_cangjie("k") == ["k"]
    assert custom_tokenizer_cangjie(" ") == [" "]

    # overpresented
    assert custom_tokenizer_cangjie("k  ") == ["k ", " "]
    assert custom_tokenizer_cangjie("mf  ogd l ") == ["mf ", " ", "ogd ", "l "]
    # edge cases
    assert custom_tokenizer_cangjie("") == []
    assert custom_tokenizer_cangjie("gjieojago") == ["gjieojago"]

    # Other languages/characters
    assert custom_tokenizer_cangjie("你好") == ["你好"]
    assert custom_tokenizer_cangjie("789$0 !@#") == ["789$0 ", "!@#"]
    # ↑↑ should raise error?


def test_custom_tokenizer_bopomofo():
    # right format
    assert custom_tokenizer_bopomofo("su3cl3a87") == ["su3", "cl3", "a87"]
    assert custom_tokenizer_bopomofo("j6vu04ru/42u4") == ["j6", "vu04", "ru/4", "2u4"]

    # underpresented
    assert custom_tokenizer_bopomofo("su3cl3a8") == ["su3", "cl3", "a8"]
    assert custom_tokenizer_bopomofo("t z") == ["t ", "z"]

    # overpresented
    assert custom_tokenizer_bopomofo("su3cl3a87 ") == ["su3", "cl3", "a87", " "]
    assert custom_tokenizer_bopomofo("nji3u.3fm06bp66") == [
        "nji3",
        "u.3",
        "fm06",
        "bp6",
        "6",
    ]

    # edge cases
    assert custom_tokenizer_bopomofo("") == []
    assert custom_tokenizer_bopomofo("  ") == [" ", " "]

    # Other languages/characters
    assert custom_tokenizer_bopomofo("你好") == ["你好"]
    # assert custom_tokenizer_cangjie("78989$290 !@#") == ["78989$290 ", "!@#"] # should raise error?


if __name__ == "__main__":
    # test_custom_tokenizer_pinyin()
    # print("All tests passed!")
    test_cases = ["mouren", "shierge", "n", "xienshi"]
    for case in test_cases:
        print(f"{case}: {custom_tokenizer_pinyin(case)}")
    # text = "mouren", "shierge"
    # print(f"{text }: {custom_tokenizer_pinyin(text)}")
