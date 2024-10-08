import re
import time

from .ime_separator import IMESeparator
from .ime_converter import ChineseIMEConverter, EnglishIMEConverter
from .candidate import CandidateWord
from .core.custom_decorators import not_implemented


def custom_tokenizer_bopomofo(text):
    if not text:
        return []
    pattern = re.compile(r"(?<=3|4|6|7| )")
    tokens = pattern.split(text)
    tokens = [token for token in tokens if token]
    if tokens[-1].find("§") != -1:
        tokens.pop()

    return tokens


def custom_tokenizer_cangjie(text):
    if not text:
        return []
    pattern = re.compile(r"(?<=[ ])")
    tokens = pattern.split(text)
    tokens = [token for token in tokens if token]
    if tokens[-1].find("§") != -1:
        tokens.pop()
    return tokens


def custom_tokenizer_pinyin(text):
    if not text:
        return []
    tokens = []
    pattern = re.compile(
        r"(?:[bpmfdtnlgkhjqxzcsyw]|[zcs]h)?(?:[aeiouv]?ng|[aeiou](?![aeiou])|[aeiou]?[aeiou]?r|[aeiou]?[aeiou]?[aeiou])"
    )
    matches = re.findall(pattern, text)
    tokens.extend(matches)
    if tokens and tokens[-1].find("§") != -1:
        tokens.pop()
    return tokens


class IMEHandler:
    def __init__(self) -> None:
        self._bopomofo_converter = ChineseIMEConverter(
            ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\bopomofo_dict_with_frequency.json"
        )
        self._cangjie_converter = ChineseIMEConverter(
            ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\cangjie_dict_with_frequency.json"
        )
        self._pinyin_converter = ChineseIMEConverter(
            ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\pinyin_dict_with_frequency.json"
        )
        self._english_converter = EnglishIMEConverter(
            ".\\multilingual_ime\\src\\keystroke_mapping_dictionary\\english_dict_with_frequency.json"
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
