import re
import time
from pathlib import Path

from .ime_separator import IMESeparator
from .ime_converter import ChineseIMEConverter, EnglishIMEConverter
from .candidate import CandidateWord
from .core.custom_decorators import not_implemented, lru_cache_with_doc


def custom_tokenizer_bopomofo(text):
    if not text:
        return []
    pattern = re.compile(r"(?<=3|4|6|7| )")
    tokens = pattern.split(text)
    tokens = [token for token in tokens if token]
    return tokens


def custom_tokenizer_cangjie(text):
    if not text:
        return []
    pattern = re.compile(r"(?<=[ ])")
    tokens = pattern.split(text)
    tokens = [token for token in tokens if token]
    return tokens


with open(
    Path(__file__).parent / "src" / "intact_pinyin.txt", "r", encoding="utf-8"
) as f:
    intact_pinyin_set = set(s for s in f.read().split("\n"))

all_pinyin_set = set(s[:i] for s in intact_pinyin_set for i in range(1, len(s) + 1))

intact_cut_pinyin_ans = {}
all_cut_pinyin_ans = {}


def custom_tokenizer_pinyin(pinyin: str) -> list[tuple[str]]:
    # Modified from https://github.com/OrangeX4/simple-pinyin.git

    @lru_cache_with_doc(maxsize=128, typed=False)
    def cut_pinyin(pinyin: str, is_intact: bool = False) -> list[tuple[str]]:
        if is_intact:
            pinyin_set = intact_pinyin_set
        else:
            pinyin_set = all_pinyin_set

        # If result is not in the word set, DP by recursion
        ans = [] if pinyin not in pinyin_set else [(pinyin,)]
        for i in range(1, len(pinyin)):
            # If pinyin[:i], is a right word, continue DP
            if pinyin[:i] in pinyin_set:
                appendices = cut_pinyin(pinyin[i:], is_intact)
                for appendix in appendices:
                    ans.append((pinyin[:i],) + appendix)
        return ans

    def cut_pinyin_with_error_correction(pinyin: str) ->  list[tuple[str]]:
        ans = {}
        for i in range(1, len(pinyin) - 1):
            key = pinyin[:i] + pinyin[i + 1] + pinyin[i] + pinyin[i + 2:]
            value = cut_pinyin(key, is_intact=True)
            if value:
                ans[key] = value
        return [p for t in ans.values() for p in t]

    intact_ans = cut_pinyin(pinyin, is_intact=True)
    not_intact_ans = cut_pinyin(pinyin, is_intact=False)
    with_error_correction_ans = cut_pinyin_with_error_correction(pinyin)
    total_ans = list(set(intact_ans + not_intact_ans + with_error_correction_ans))
    total_ans = sorted(total_ans, key=lambda x: len(x))
    print(total_ans)
    return total_ans


def custom_tokenizer_english(text):
    if not text:
        return []
    tokens = re.findall(r"\w+|[^\w\s]|\s", text)
    return tokens


class IMEHandler:
    def __init__(self) -> None:
        self._bopomofo_converter = ChineseIMEConverter(
            Path(__file__).parent
            / "src"
            / "keystroke_mapping_dictionary"
            / "bopomofo_dict_with_frequency.json"
        )
        self._cangjie_converter = ChineseIMEConverter(
            Path(__file__).parent
            / "src"
            / "keystroke_mapping_dictionary"
            / "cangjie_dict_with_frequency.json"
        )
        self._pinyin_converter = ChineseIMEConverter(
            Path(__file__).parent
            / "src"
            / "keystroke_mapping_dictionary"
            / "pinyin_dict_with_frequency.json"
        )
        self._english_converter = EnglishIMEConverter(
            Path(__file__).parent
            / "src"
            / "keystroke_mapping_dictionary"
            / "english_dict_with_frequency.json"
        )
        self._separator = IMESeparator(use_cuda=False)

    def _get_candidate_words(
        self, keystroke: str, prev_context: str = ""
    ) -> list[list[CandidateWord]]:
        separate_possibilities = self._separator.separate(keystroke)
        candidate_sentences = []
        for separate_way in separate_possibilities:
            candidate_sentences.append(self._construct_sentence(separate_way))
        assert len(separate_possibilities) == len(
            candidate_sentences
        ), "Length of separate_possibilities and candidate_sentences should be the same"

        candidate_sentences = sorted(
            candidate_sentences, key=lambda x: x["total_distance"]
        )
        return candidate_sentences

    def _construct_sentence(self, separate_way) -> list[list[CandidateWord]]:
        logical_sentence = []
        for method, keystroke in separate_way:
            if method == "bopomofo":
                tokens = custom_tokenizer_bopomofo(keystroke)
                for token in tokens:
                    logical_sentence.append(
                        [
                            g.set_method("bopomofo")
                            for g in self._bopomofo_converter.get_candidates(token)
                        ]
                    )
            elif method == "cangjie":
                tokens = custom_tokenizer_cangjie(keystroke)
                for token in tokens:
                    logical_sentence.append(
                        [
                            g.set_method("cangjie")
                            for g in self._cangjie_converter.get_candidates(token)
                        ]
                    )
            elif method == "pinyin":
                tokens = custom_tokenizer_pinyin(keystroke)
                for token in tokens:
                    logical_sentence.append(
                        [
                            g.set_method("pinyin")
                            for g in self._pinyin_converter.get_candidates(token)
                        ]
                    )
            elif method == "english":
                tokens = keystroke.split(" ")
                for token in tokens:
                    logical_sentence.append(
                        [
                            g.set_method("english")
                            for g in self._english_converter.get_candidates(token)
                        ]
                    )
            else:
                raise ValueError("Invalid method: " + method)

        logical_sentence = [
            logical_word for logical_word in logical_sentence if len(logical_word) > 0
        ]
        sum_distance = sum(
            [logical_word[0].distance for logical_word in logical_sentence]
        )

        return {"total_distance": sum_distance, "sentence": logical_sentence}

    def _construct_sentence_to_words(self, logical_sentence) -> list[list[str]]:
        sentences = []
        for logical_sentence in logical_sentence:
            sentence = [candidate_word.word for candidate_word in logical_sentence]
            sentences.append(sentence)
        return sentences

    @not_implemented
    def _greedy_phrase_search(self, logical_sentence, prev_context):
        pass

    def get_candidate(self, keystroke: str, prev_context: str = "") -> list[str]:
        result = self._get_candidate_words(keystroke, prev_context)
        best_logical_sentence = result[0]["sentence"]
        return self._construct_sentence_to_words(best_logical_sentence)


if __name__ == "__main__":
    context = ""
    user_keystroke = "t g3bjo4dk4apple wathc"
    start_time = time.time()
    my_IMEHandler = IMEHandler()
    print("Initialization time: ", time.time() - start_time)
    avg_time, num_of_test = 0, 0
    while True:
        user_keystroke = input("Enter keystroke: ")
        num_of_test += 1
        start_time = time.time()
        result = my_IMEHandler.get_candidate(user_keystroke, context)
        end_time = time.time()
        avg_time = (avg_time * (num_of_test - 1) + end_time - start_time) / num_of_test
        print(f"Inference time: {time.time() - start_time}, avg time: {avg_time}")
        print(result)
