
from tqdm import tqdm
from multilingual_ime.ime_handler import custom_tokenizer_pinyin
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
            converter_keystroke_result = KeyStrokeConverter.convert(plain_text, "pinyin")
            con_tokens = custom_tokenizer_pinyin(converter_keystroke_result)
            direct_tokens = custom_tokenizer_pinyin(keystroke_data_text)

            joined_con_tokens = "".join(con_tokens)
            if len(joined_con_tokens) != len(converter_keystroke_result):
                print(f"Error Losing characters:\n"
                      f"plaintext: {plain_text}\n"
                      f"con: {converter_keystroke_result}\n"
                      f"joint tokens: {joined_con_tokens}\n"
                      f"direct_tokens:{direct_tokens}\n\n")
                
            
            if con_tokens != direct_tokens:
                print(f"Error: Tokenization mismatch \n"
                      f"plaintext: {plain_text}\n"
                      f"keystroke data text:{keystroke_data_text}/{direct_tokens}\n"
                      f"con:{converter_keystroke_result}/{con_tokens}\n"
                      f"direct_tokens:{direct_tokens}\n\n")
                
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
            


if __name__ == "__main__":
    # test_custom_tokenizer_pinyin()
    # print("All tests passed!")
    test_cases = ["mouren", "shierge", "n", "xienshi"]
    for case in test_cases:
        print(f"{case}: {custom_tokenizer_pinyin(case)}")
    # text = "mouren", "shierge"
    # print(f"{text }: {custom_tokenizer_pinyin(text)}")