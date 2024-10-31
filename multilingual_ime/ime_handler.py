import time
import logging

from .candidate import Candidate
from .core.custom_decorators import lru_cache_with_doc
from .ime import BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME, PINYIN_IME
from .ime import IMEFactory
from .trie import modified_levenshtein_distance


class IMEHandler:
    def __init__(self, verbose_mode: bool = False) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose_mode else logging.WARNING)
        self.ime_list = [BOPOMOFO_IME, CANGJIE_IME, PINYIN_IME, ENGLISH_IME]
        self.ime_handlers = {ime: IMEFactory.create_ime(ime) for ime in self.ime_list}

    @lru_cache_with_doc(maxsize=128)
    def get_token_candidates(self, token: str) -> list[Candidate]:
        candidates = []

        for ime_type in self.ime_list:
            if self.ime_handlers[ime_type].is_valid_token(token):
                result = self.ime_handlers[ime_type].get_token_candidates(token)
                candidates.extend(
                    [
                        Candidate(
                            word,
                            key,
                            frequency,
                            token,
                            modified_levenshtein_distance(key, token),
                            ime_type,
                        )
                        for key, word, frequency in result
                    ]
                )

        candidates = sorted(candidates, key=lambda x: x.distance)
        assert len(candidates) > 0, f"No candidates found for token '{token}'"
        return candidates

    def get_candidate_sentences(self, keystroke: str, context: str = "") -> list[dict]:
        start_time = time.time()
        token_pool = self._get_token_pool(keystroke)
        self.logger.info(f"Token pool time: {time.time() - start_time}")
        self.logger.info(f"Token pool: {token_pool}")
        token_pool = set(
            [token for token in token_pool if self._is_valid_token(token)]
        )  # Filter out invalid token
        self.logger.info(f"Filter out invalid token time: {time.time() - start_time}")
        self.logger.info(f"Filtered token pool: {token_pool}")
        possible_sentences = self._reconstruct_sentence(keystroke, token_pool)
        self.logger.info(f"Reconstruct sentence time: {time.time() - start_time}")
        self.logger.info(f"Possible sentences: {possible_sentences}")
        result = []
        for sentence in possible_sentences:
            ans_sentence_distance = 0
            for token in sentence:
                assert (
                    token in token_pool
                ), f"Token '{token}' not in token pool {token_pool}"
                ans_sentence_distance += self._closest_word_distance(token)
            result.append({"sentence": sentence, "distance": ans_sentence_distance})

        result = sorted(result, key=lambda x: x["distance"])

        # Filter out none best result
        filter_out_none_best_result = True
        if filter_out_none_best_result:
            best_distance = result[0]["distance"]
            result = [r for r in result if r["distance"] <= best_distance]

        return result

    def get_best_sentence(self, keystroke: str, context: str = "") -> dict:
        best_candidate_sentences = self.get_candidate_sentences(keystroke, context)[0][
            "sentence"
        ]
        output = ""
        for token in best_candidate_sentences:
            output += self.get_token_candidates(token)[0].word

        return output

    def _reconstruct_sentence(self, keystroke: str, token_pool: set) -> list[list[str]]:
        """
        Reconstruct the sentence from the keystroke.

        Args:

                keystroke (str): The keystroke to search for
        Returns:

                list: A list of **tuples (token, method)** containing the possible tokens

        """

        def dp_search(keystroke: str) -> list[list[str]]:
            if not keystroke:
                return [[]]

            ans = []
            for token_str in token_pool:
                if keystroke.startswith(token_str):
                    ans.extend(
                        [
                            [token_str] + sub_ans
                            for sub_ans in dp_search(keystroke[len(token_str) :])
                        ]
                    )

            if keystroke in token_pool:
                ans.append([keystroke])
            return ans

        result = dp_search(keystroke)
        unique_result = list(map(list, set(map(tuple, result))))
        return unique_result

    def _get_token_pool(self, keystroke: str) -> set[tuple[str, str]]:
        """
        Tokenize string of keystroke into token pool.

        Args:

            keystroke (str): The keystroke to search for
        Returns:
            set: A set of **token(str)** containing the possible tokens

        """
        token_pool = set()

        for ime_type in self.ime_list:
            token_ways = self.ime_handlers[ime_type].tokenize(keystroke)
            for ways in token_ways:
                for token in ways:
                    token_pool.add(token)
        return token_pool

    @lru_cache_with_doc(maxsize=128)
    def _is_valid_token(self, token: str) -> bool:
        if not token:
            return False

        for ime_type in self.ime_list:
            if self.ime_handlers[ime_type].is_valid_token(token):
                return True
        return False

    @lru_cache_with_doc(maxsize=128)
    def _closest_word_distance(self, token: str) -> int:
        min_distance = float("inf")

        for ime_type in self.ime_list:
            if not self.ime_handlers[ime_type].is_valid_token(token):
                continue

            method_distance = self.ime_handlers[ime_type].closest_word_distance(token)
            min_distance = min(min_distance, method_distance)
        return min_distance


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
        # result = my_IMEHandler.get_candidate_sentences(user_keystroke, context)
        # result = my_IMEHandler._get_token_candidates(user_keystroke)
        result = my_IMEHandler.get_best_sentence(user_keystroke, context)
        end_time = time.time()
        avg_time = (avg_time * (num_of_test - 1) + end_time - start_time) / num_of_test
        print(f"Inference time: {time.time() - start_time}, avg time: {avg_time}")
        print(result)
