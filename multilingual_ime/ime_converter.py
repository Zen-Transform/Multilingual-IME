import json
from abc import ABC, abstractmethod
from pathlib import Path

from .trie import Trie
from .candidate import CandidateWord


class IMEConverter: ...


class ChineseIMEConverter(IMEConverter): ...


class EnglishIMEConverter(IMEConverter): ...


class IMEConverter(ABC):
    @abstractmethod
    def __init__(self, data_dict_path: str):
        self.trie = Trie()
        try:
            keystroke_mapping_dict = json.load(
                open(data_dict_path, "r", encoding="utf-8")
            )
            if keystroke_mapping_dict is not None:
                for key, value in keystroke_mapping_dict.items():
                    Candidate_words = [
                        CandidateWord(
                            word=element[0], keystrokes=key, word_frequency=element[1]
                        )
                        for element in value
                    ]
                    for candidate in Candidate_words:
                        self.trie.insert(key, candidate)
        except Exception as e:
            print(f"Error loading data dictionary from {data_dict_path}")
            print(e)

    @abstractmethod
    def get_candidates(self):
        pass


class ChineseIMEConverter(IMEConverter):
    def __init__(self, data_dict_path: str):
        super().__init__(data_dict_path)

    def get_candidates(self, key_stroke_query: str) -> list[CandidateWord]:
        candidates = self.trie.find_closest_match(key_stroke_query)
        assert len(candidates) > 0, f"No candidate found for {key_stroke_query}"

        word_candidates = []
        for candidate in candidates:
            for candidate_word in candidate["value"]:
                candidate_word.distance = candidate["distance"]
                candidate_word.user_key = key_stroke_query
                word_candidates.append(candidate_word)
        return word_candidates


class EnglishIMEConverter(IMEConverter):
    def __init__(self, data_dict_path: str):
        super().__init__(data_dict_path)

    def get_candidates(self, key_stroke_query: str) -> list[CandidateWord]:
        key_stroke_query_lower_case = (
            key_stroke_query.lower()
        )  # english specail modification: remove case sensitivity

        candidates = self.trie.find_closest_match(key_stroke_query_lower_case)
        assert (
            len(candidates) > 0
        ), f"No candidate found for {key_stroke_query_lower_case}"

        word_candidates = []
        for candidate in candidates:
            for candidate_word in candidate["value"]:
                new_word = CandidateWord(
                    candidate_word.word, key_stroke_query, candidate_word.word_frequency
                )
                new_word.distance = candidate["distance"]
                new_word.user_key = key_stroke_query
                if new_word.word.lower() == key_stroke_query_lower_case:
                    new_word.word = key_stroke_query
                word_candidates.append(new_word)
        return word_candidates


if __name__ == "__main__":
    my_bopomofo_IMEConverter = ChineseIMEConverter(
        Path(__file__).parent
        / "src"
        / "keystroke_mapping_dictionary"
        / "bopomofo_dict_with_frequency.json"
    )
    my_cangjie_IMEConverter = ChineseIMEConverter(
        Path(__file__).parent
        / "src"
        / "keystroke_mapping_dictionary"
        / "cangjie_dict_with_frequency.json"
    )
    my_pinyin_IMEConverter = ChineseIMEConverter(
        Path(__file__).parent
        / "src"
        / "keystroke_mapping_dictionary"
        / "pinyin_dict_with_frequency.json"
    )
    my_english_IMEConverter = EnglishIMEConverter(
        Path(__file__).parent
        / "src"
        / "keystroke_mapping_dictionary"
        / "english_dict_with_frequency.json"
    )

    for candidate_word in my_english_IMEConverter.get_candidates("APPLE"):
        print(candidate_word.to_dict())
