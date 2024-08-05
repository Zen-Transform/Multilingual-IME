import re

from multilingual_ime.ime_separator import IMESeparator
from multilingual_ime.ime_converter import ChineseIMEConverter, EnglishIMEConverter
from multilingual_ime.candidate import CandidateWord


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

    def get_candidate_words(
        self, keystroke: str, prev_context: str = ""
    ) -> list[list[CandidateWord]]:
        separate_possibilities = self._separator.separate(keystroke)
        sentence_possibilities = []
        for separate_way in separate_possibilities:
            sentence_possibilities.append(self._construct_sentence(separate_way))
        assert len(separate_possibilities) == len(
            sentence_possibilities
        ), "Length of separate_possibilities and sentence_possibilities should be the same"

        sentence_possibilities = sorted(
            sentence_possibilities, key=lambda x: x["total_distance"]
        )
        return sentence_possibilities

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


if __name__ == "__main__":
    my_IMEHandler = IMEHandler()
    context = ""
    user_keystroke = "su3cl3good night"
    result = my_IMEHandler.get_candidate_words(user_keystroke, context)
    for i, my_dict in enumerate(result, 1):
        print(f"Result {i}:")
        print(f"Total distance: {my_dict['total_distance']}")
        sentence = my_dict["sentence"]
        for candidate_words in sentence:
            print(
                " - "
                + " ".join([candidate_word.word for candidate_word in candidate_words])
            )
        print()
