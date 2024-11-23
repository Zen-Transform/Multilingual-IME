import time
import logging
from pathlib import Path

import jieba

from .candidate import Candidate
from .core.custom_decorators import lru_cache_with_doc
from .ime import BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME, PINYIN_IME
from .ime import IMEFactory
from .phrase_db import PhraseDataBase
from .trie import modified_levenshtein_distance

CHINESE_PHRASE_DB_PATH = Path(__file__).parent / "src" / "chinese_phrase.db"
USER_PHRASE_DB_PATH = Path(__file__).parent / "src" / "user_phrase.db"


class IMEHandler:
    def __init__(self, verbose_mode: bool = False) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose_mode else logging.WARNING)
        self.logger.addHandler(logging.StreamHandler())
        self.ime_list = [BOPOMOFO_IME, CANGJIE_IME, PINYIN_IME, ENGLISH_IME]
        self.ime_handlers = {ime: IMEFactory.create_ime(ime) for ime in self.ime_list}
        self._chinese_phrase_db = PhraseDataBase(CHINESE_PHRASE_DB_PATH)
        self._user_phrase_db = PhraseDataBase(USER_PHRASE_DB_PATH)
        self.valid_token_pool = set()

        self.auto_phrase_learning = True

    @lru_cache_with_doc(maxsize=128)
    def get_token_candidates(self, token: str) -> list[Candidate]:
        """
        Get the possible candidates of the token from all IMEs.

        Args:
            token (str): The token to search for
        Returns:
            list: A list of **Candidate** containing the possible candidates
        """

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

        if len(candidates) == 0:
            self.logger.info(f"No candidates found for token '{token}'")
            return [Candidate(token, token, 0, token, 0, "NO_IME")]

        candidates = sorted(candidates, key=lambda x: x.distance)
        return candidates

    def get_token_candidate_words(self, token: str) -> list[str]:
        """
        Get the possible candidate words of the token from all IMEs.

        Args:
            token (str): The token to search for
        Returns:
            list: A list of **str** containing the possible candidate words
        """

        candidates = self.get_token_candidates(token)
        return [candidate.word for candidate in candidates]

    def get_candidate_sentences(self, keystroke: str, context: str = "") -> list[dict]:
        """
        Get the possible combination of tokens that have the least distance to the keystroke.

        Args:
            keystroke (str): The keystroke to search for
            context (str): The context to search for

        Returns:
            list: A list of **dict** containing the possible sentences and their distance.
                **dict["sentence"]**: list of str, the possible sentence.
                **dict["distance"]**: int, the distance of the sentence.
        """

        start_time = time.time()
        token_pool = self._get_token_pool(keystroke)
        self.logger.info(f"Token pool time: {time.time() - start_time}")
        self.logger.info(f"Token pool: {token_pool}")
        
        start_time = time.time()
        self.valid_token_pool = set(
            [token for token in token_pool if self._is_valid_token(token)]
        )  # Filter out invalid token
        self.logger.info(f"Filter out invalid token time: {time.time() - start_time}")
        self.logger.info(f"Filtered token pool: {self.valid_token_pool}")
        
        start_time = time.time()
        possible_sentences = self._reconstruct_sentence(keystroke, self.valid_token_pool)
        if not possible_sentences:  # Solve for the case where there is no possible sentence in the valid token pool
            possible_sentences = self._reconstruct_sentence(keystroke, token_pool)
        self.logger.info(f"Reconstruct sentence time: {time.time() - start_time}")
        self.logger.info(f"Possible sentences: {possible_sentences}")


        start_time = time.time()
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
        self.logger.info(f"Get candidate/search sentences distance time: {time.time() - start_time}")

        # Filter out none best result
        filter_out_none_best_result = True
        if filter_out_none_best_result:
            best_distance = result[0]["distance"]
            result = [r for r in result if r["distance"] <= best_distance]

        return result

    def get_best_sentence(self, keystroke: str, context: str = "") -> str:
        """
        Get the best sentence in words based on the keystroke and context.

        Args:
            keystroke (str): The keystroke to search for
            context (str): The context to search for
        Returns:
            str: The best sentence of words

        """

        def solve_sentence(pre_word: str, bestsentence_tokens: list[list[Candidate]]):

            def recursive(best_sentence_tokens: list[list[Candidate]]) -> str:
                if not best_sentence_tokens:
                    return []

                related_phrases = []
                for candidate in best_sentence_tokens[0]:
                    related_phrases.extend(
                        self._chinese_phrase_db.get_phrase_with_prefix(candidate.word)
                    )
                    related_phrases.extend(
                        self._user_phrase_db.get_phrase_with_prefix(candidate.word)
                    )
                
                related_phrases = [phrase[0] for phrase in related_phrases]
                related_phrases = sorted(
                    related_phrases, key=lambda x: len(x), reverse=True
                )
                related_phrases = [
                    phrase
                    for phrase in related_phrases
                    if len(phrase) <= len(best_sentence_tokens)
                ]

                for phrase in related_phrases:
                    correct_phrase = True
                    for i, char in enumerate(phrase):
                        if char not in [
                            candidate.word for candidate in best_sentence_tokens[i]
                        ]:
                            correct_phrase = False
                            break

                    if correct_phrase:
                        return [phrase] + recursive(best_sentence_tokens[len(phrase) :])

                return [best_sentence_tokens[0][0].word] + recursive(
                    best_sentence_tokens[1:]
                )

            return recursive(bestsentence_tokens)

        best_candidate_sentences = self.get_candidate_sentences(keystroke, context)[0][
            "sentence"
        ]
        best_sentence_tokens = [
            self.get_token_candidates(token) for token in best_candidate_sentences
        ]

        pre_word = context[-1] if context else ""
        start_time = time.time()
        result = "".join(solve_sentence(pre_word, best_sentence_tokens))
        self.logger.info(f"Get Best sentence time: {time.time() - start_time}")

        return result

    def _reconstruct_sentence(self, keystroke: str, token_pool: set) -> list[list[str]]:
        """
        Reconstruct the sentence back to the keystroke by searching all the
        possible combination of tokens in the token pool.

        Args:
            keystroke (str): The keystroke to search for
        Returns:
            list: A list of **list of str** containing the possible sentences constructed from the token pool
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

    def _get_token_pool(self, keystroke: str) -> set[str]:
        """
        Tokenize string of keystroke by all IMEs and return the possible tokens in a set.

        Args:
            keystroke (str): The keystroke to tokenize
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
        """
        Check if the token is valid, i.e. it is valid in at least one IME.

        Args:
            token (str): The token to check
        Returns:
            bool: True if the token is valid, False otherwise
        """

        if not token:
            return False

        for ime_type in self.ime_list:
            if self.ime_handlers[ime_type].is_valid_token(token):
                return True
        return False

    @lru_cache_with_doc(maxsize=128)
    def _closest_word_distance(self, token: str) -> int:
        """
        Get the word distance to the closest word from all IMEs.

        Args:
            token (str): The token to search for
        Returns:
            int: The distance to the closest word
        """
        min_distance = float("inf")

        if token not in self.valid_token_pool:
            return min_distance
        
        for ime_type in self.ime_list:
            if not self.ime_handlers[ime_type].is_valid_token(token):
                continue

            method_distance = self.ime_handlers[ime_type].closest_word_distance(token)
            min_distance = min(min_distance, method_distance)
        return min_distance

    def update_user_phrase_db(self, text: str) -> None:
        """
        Update the user phrase database with the given phrase and frequency.

        Args:
            phrase (str): The phrase to update
            frequency (int): The frequency of the phrase
        """

        for phrase in jieba.lcut(text, cut_all=False):
            if not self._user_phrase_db.getphrase(phrase):
                self._user_phrase_db.insert(phrase, 1)
            else:
                self._user_phrase_db.increment_frequency(phrase)


if __name__ == "__main__":
    context = ""
    user_keystroke = "t g3bjo4dk4apple wathc"
    start_time = time.time()
    my_IMEHandler = IMEHandler(verbose_mode=True)
    print("Initialization time: ", time.time() - start_time)
    avg_time, num_of_test = 0, 0
    while True:
        user_keystroke = input("Enter keystroke: ")
        num_of_test += 1
        start_time = time.time()
        result = my_IMEHandler.get_candidate_sentences(user_keystroke, context)
        print("result", result)
        # result = my_IMEHandler.get_token_candidates(user_keystroke)
        result = my_IMEHandler.get_best_sentence(user_keystroke, context)
        end_time = time.time()
        avg_time = (avg_time * (num_of_test - 1) + end_time - start_time) / num_of_test
        print(f"Inference time: {time.time() - start_time}, avg time: {avg_time}")
        print(result)
