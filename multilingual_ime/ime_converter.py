import json
from abc import ABC, abstractmethod

from .trie import Trie
from .candidate import CandidateWord


class IMEConverter: ...


class ChineseIMEConverter(IMEConverter): ...


class EnglishIMEConverter(IMEConverter): ...


class IMEConverter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_candidates(self):
        pass


class ChineseIMEConverter(IMEConverter):
    def __init__(self, data_dict_path: str):
        self.trie = Trie(json.load(open(data_dict_path, "r", encoding="utf-8")))

    def get_candidates(self, key_stroke_query: str) -> list[CandidateWord]:
        candidates = self.trie.findClosestMatches(key_stroke_query)
        min_distance = min([candidate["distance"] for candidate in candidates])
        candidates = [
            candidate
            for candidate in candidates
            if candidate["distance"] == min_distance
        ]  # filter out candidates with distance greater than min_distance
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
        self.trie = Trie(json.load(open(data_dict_path, "r", encoding="utf-8")))

    def get_candidates(self, key_stroke_query: str) -> list[CandidateWord]:
        key_stroke_query_lower_case = key_stroke_query.lower()

        candidates = self.trie.findClosestMatches(key_stroke_query_lower_case)
        min_distance = min([candidate["distance"] for candidate in candidates])
        candidates = [
            candidate
            for candidate in candidates
            if candidate["distance"] == min_distance
        ]  # filter out candidates with distance greater than min_distance
        assert (
            len(candidates) > 0
        ), f"No candidate found for {key_stroke_query_lower_case}"

        word_candidates = []
        for candidate in candidates:
            for candidate_word in candidate["value"]:
                candidate_word.distance = candidate["distance"]
                candidate_word.user_key = key_stroke_query
                word_candidates.append(candidate_word)
        return word_candidates


if __name__ == "__main__":
    my_bopomofo_IMEConverter = ChineseIMEConverter(
        ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\bopomofo_dict_with_frequency.json"
    )
    my_cangjie_IMEConverter = ChineseIMEConverter(
        ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\cangjie_dict_with_frequency.json"
    )
    my_pinyin_IMEConverter = ChineseIMEConverter(
        ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\pinyin_dict_with_frequency.json"
    )
    my_english_IMEConverter = EnglishIMEConverter(
        ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\english_dict_with_frequency.json"
    )

    for candidate_word in my_english_IMEConverter.get_candidates("APPLE9090909090"):
        print(candidate_word.to_dict())
