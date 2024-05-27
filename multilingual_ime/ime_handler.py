
from typing import Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import time

from multilingual_ime.ime_separator import IMESeparator
from multilingual_ime.ime_converter import IMEConverter
from multilingual_ime.ime_converter import EnglishIMEConverter
from SVMClassification import (
    custom_tokenizer_bopomofo,
    custom_tokenizer_cangjie,
    custom_tokenizer_pinyin,
)


class IMEHandler():
    def __init__(self) -> None:
        self._bopomofo_converter = IMEConverter(".\\keystroke_mapping_dictionary\\bopomofo_dict_with_frequency.json")
        self._cangjie_converter = IMEConverter(".\\keystroke_mapping_dictionary\\cangjie_dict_with_frequency.json")
        self._pinyin_converter = IMEConverter(".\\keystroke_mapping_dictionary\\pinyin_dict_with_frequency.json")
        self._english_converter = EnglishIMEConverter(".\\keystroke_mapping_dictionary\\english_dict_with_frequency.json")
        self._separator = IMESeparator()
        self._language_model = None
        self._load_language_model("all-MiniLM-L6-v2")

    def _load_language_model(self, model_name: str) -> SentenceTransformer:
        start_time = time.time()
        print(f"Loading model...")
        self._language_model = SentenceTransformer(model_name)
        print("Model loaded successfully!", time.time() - start_time, "seconds")

    def get_candidate_words(self, keystroke: str, prev_context: str = "") -> list[dict[str, Union[int, list[dict[str, Union[str, int]]]]]]:
        separate_possibilities = self._separator.separate(keystroke)
        output = []
        for separate_way in separate_possibilities:
            logical_sentence = []
            for method, keystroke in separate_way:
                if method == "bopomofo":
                    tokens = custom_tokenizer_bopomofo(keystroke)
                    for token in tokens:
                        logical_sentence.append([{**g, "method": "bopomofo"} for g in self._bopomofo_converter.get_candidates(token, 1, 2)])
                elif method == "cangjie":
                    tokens = custom_tokenizer_cangjie(keystroke)
                    for token in tokens:
                        logical_sentence.append([{**g, "method": "cangjie"} for g in self._cangjie_converter.get_candidates(token, 1, 2)])
                elif method == "pinyin":
                    tokens = custom_tokenizer_pinyin(keystroke)
                    for token in tokens:
                        logical_sentence.append([{**g, "method": "pinyin"} for g in self._pinyin_converter.get_candidates(token, 1, 2)])
                elif method == "english":
                    tokens = keystroke.split(" ")
                    for token in tokens:
                        logical_sentence.append([{**g, "method": "english"} for g in self._english_converter.get_candidates(token, 1, 2)])
                else:
                    raise ValueError("Invalid method: " + method)
                
            logical_sentence = [g for g in logical_sentence if len(g) > 0]
            # logical_sentence.sort(key=lambda x: x[0]["distance"])  # bad
            # print("logical_sentence", logical_sentence)
            sum_distance = sum([g[0]["distance"] for g in logical_sentence])
            
            output.append({
                "total_distance": sum_distance,
                "sentence": logical_sentence
                })
        
        output = sorted(output, key=lambda x: x["total_distance"])
        min_total_distance = output[0]["total_distance"]
        output = [g for g in output if g["total_distance"] <= min_total_distance]

        if len(prev_context) > 0:
            encoded_context = self._language_model.encode(prev_context)

            for possible_sentence_group in output:
                for logic_token in possible_sentence_group["sentence"]:
                    for possible_word in logic_token:
                        possible_word["similarity_score"] = float(cos_sim(self._language_model.encode(possible_word["word"]), encoded_context).item())  # bad
                    logic_token.sort(key=lambda x: x["similarity_score"], reverse=True)
        return output


if __name__ == "__main__":
    my_IMEHandler = IMEHandler()
    while True:
        context = input("Input Context: ")
        user_keystroke = input("Input Keystroke: ")
        result = my_IMEHandler.get_candidate_words(user_keystroke, prev_context=context)
        # print(result)
        
        with open("output.json", "w", encoding="utf-8") as f:
            f.write(str(result))
        print("===")

