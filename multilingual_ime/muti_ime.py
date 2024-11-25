import time
import logging
from pathlib import Path

import jieba

from .candidate import Candidate
from .core.custom_decorators import lru_cache_with_doc, deprecated
from .ime import BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME, PINYIN_IME
from .ime import IMEFactory
from .phrase_db import PhraseDataBase
from .trie import modified_levenshtein_distance

from colorama import Fore, Style
import threading

from .ime import (
    BOPOMOFO_VALID_KEYSTROKE_SET,
    ENGLISH_VALID_KEYSTROKE_SET,
    PINYIN_VALID_KEYSTROKE_SET,
    CANGJIE_VALID_KEYSTROKE_SET,
)

TOTAL_VALID_KEYSTROKE_SET = (
    BOPOMOFO_VALID_KEYSTROKE_SET.union(ENGLISH_VALID_KEYSTROKE_SET)
    .union(PINYIN_VALID_KEYSTROKE_SET)
    .union(CANGJIE_VALID_KEYSTROKE_SET)
)

CHINESE_PHRASE_DB_PATH = Path(__file__).parent / "src" / "chinese_phrase.db"
USER_PHRASE_DB_PATH = Path(__file__).parent / "src" / "user_phrase.db"

WITH_COLOR = True


class KeyEventHandler:
    def __init__(self, verbose_mode: bool = False) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose_mode else logging.WARNING)
        self.logger.addHandler(logging.StreamHandler())
        self.ime_list = [BOPOMOFO_IME, CANGJIE_IME, PINYIN_IME, ENGLISH_IME]
        self.ime_handlers = {ime: IMEFactory.create_ime(ime) for ime in self.ime_list}
        self._chinese_phrase_db = PhraseDataBase(CHINESE_PHRASE_DB_PATH)
        self._user_phrase_db = PhraseDataBase(USER_PHRASE_DB_PATH)

        self.pre_context = ""

        # Config Settings
        self.AUTO_PHRASE_LEARN = True
        self.SELECTION_PAGE_SIZE = 5

        # Other State Variables
        self._last_key_event = None
        self._timer = None

        # State Variables
        self._token_pool_set = set()
        self.have_selected = False

        self.freezed_index = 0
        self.freezed_token_sentence = []
        self.freezed_composition_words = []

        self.unfreezed_index = 0
        self.unfreezed_keystrokes = ""
        self.unfreezed_token_sentence = []
        self.unfreezed_composition_words = []

        # Selection States
        self.in_selection_mode = False
        self.selection_index = 0
        self.candidate_word_list = []

    def _reset_states(self) -> None:
        self._token_pool_set = set()
        self.have_selected = False

        self.freezed_index = 0
        self.freezed_token_sentence = []
        self.freezed_composition_words = []

        self.unfreezed_index = 0
        self.unfreezed_keystrokes = ""
        self.unfreezed_token_sentence = []
        self.unfreezed_composition_words = []

        self._reset_selection_states()

    def _reset_selection_states(self) -> None:
        self.in_selection_mode = False
        self.selection_index = 0
        self.candidate_word_list = []

    def _unfreeze_to_freeze(self) -> None:
        self.freezed_token_sentence = self.total_token_sentence
        self.freezed_composition_words = self.total_composition_words
        self.freezed_index = self.composition_index

        self.unfreezed_keystrokes = ""
        self.unfreezed_token_sentence = []
        self.unfreezed_composition_words = []
        self.unfreezed_index = 0  # Reset the index should be the last step

    @property
    def token_pool(self) -> list[tuple[str, int]]:
        return list(self._token_pool_set)

    @property
    def total_composition_words(self) -> list[str]:
        return (
            self.freezed_composition_words[: self.freezed_index]
            + self.unfreezed_composition_words
            + self.freezed_composition_words[self.freezed_index :]
        )

    @property
    def total_token_sentence(self) -> list[str]:
        return (
            self.freezed_token_sentence[: self.freezed_index]
            + self.unfreezed_token_sentence
            + self.freezed_token_sentence[self.freezed_index :]
        )

    @property
    def composition_index(self) -> int:
        return self.freezed_index + self.unfreezed_index

    def get_composition_string(self) -> str:
        return "".join(self.total_composition_words)

    def handle_key(self, key: str) -> None:
        special_keys = ["enter", "left", "right", "down", "up", "esc"]
        if key in special_keys:
            if self.in_selection_mode:
                if key == "down":
                    if self.selection_index < len(self.candidate_word_list) - 1:
                        self.selection_index += 1
                elif key == "up":
                    if self.selection_index > 0:
                        self.selection_index -= 1
                elif (
                    key == "enter"
                ):  # Overwrite the composition string & reset selection states
                    self.have_selected = True
                    selected_word = self.candidate_word_list[self.selection_index]
                    self.freezed_composition_words[self.composition_index - 1] = (
                        selected_word
                    )
                    # ! Recaculate the index
                    self.freezed_index = self.freezed_index + len(selected_word) - 1
                    self._reset_selection_states()
                elif key == "left":  # Open side selection ?
                    pass
                elif key == "right":
                    pass
                elif key == "esc":
                    self._reset_selection_states()
                else:
                    print(f"Invalid Special key: {key}")

                return
            else:
                if key == "enter":
                    print("Ouputing:", self.get_composition_string())
                    self._reset_states()
                elif key == "left":
                    self._unfreeze_to_freeze()
                    if self.freezed_index > 0:
                        self.freezed_index -= 1
                elif key == "right":
                    self._unfreeze_to_freeze()
                    if self.freezed_index < len(self.total_composition_words):
                        self.freezed_index += 1
                elif key == "down":  # Enter selection mode
                    self._unfreeze_to_freeze()
                    if (
                        len(self.total_token_sentence) > 0
                        and self.composition_index > 0
                    ):
                        self.in_selection_mode = True
                        token = self.total_token_sentence[self.composition_index - 1]
                        self.candidate_word_list = self.get_token_candidate_words(token)
                elif key == "esc":
                    self._reset_states()
                else:
                    print(f"Invalid Special key: {key}")

                return
        else:
            if key == "backspace":
                if self.unfreezed_index > 0:
                    self.unfreezed_keystrokes = self.unfreezed_keystrokes[:-1]
                    self.unfreezed_index -= 1
                else:
                    if self.freezed_index > 0:
                        self.freezed_composition_words = self.freezed_composition_words[: self.freezed_index - 1] + self.freezed_composition_words[self.freezed_index:]
                        self.freezed_index -= 1
                        return
            elif key == "space":
                self.unfreezed_keystrokes += " "
                self.unfreezed_composition_words += key
                self.unfreezed_index += 1
            elif key in TOTAL_VALID_KEYSTROKE_SET:
                self.unfreezed_keystrokes += key
                self.unfreezed_composition_words += key
                self.unfreezed_index += 1
            else:
                print(f"Invalid key: {key}")
                return

            start_time = time.time()
            self._update_token_pool()
            self.logger.info(f"Updated token pool: {time.time() - start_time}")
            self.logger.info(f"Token pool: {self.token_pool}")

            start_time = time.time()
            possible_sentences = self._reconstruct_sentence(self.unfreezed_keystrokes)
            self.logger.info(f"Reconstructed sentence: {time.time() - start_time}")
            self.logger.info(f"Reconstructed sentences: {possible_sentences}")

            start_time = time.time()
            possible_sentences = self._filter_possible_sentences_by_distance(
                possible_sentences
            )
            possible_sentences = self._get_best_sentence(possible_sentences)
            self.unfreezed_token_sentence = possible_sentences
            self.logger.info(f"Filtered sentence: {time.time() - start_time}")
            self.logger.info(f"Filtered sentences: {possible_sentences}")

            start_time = time.time()
            self.unfreezed_composition_words = self._token_sentence_to_word_sentence(
                possible_sentences
            )
            self.logger.info(f"Token to word sentence: {time.time() - start_time}")
            self.logger.info(
                f"Token to word sentences: {self.unfreezed_composition_words}"
            )

            self.unfreezed_index = len(self.unfreezed_composition_words)

            return

    def _update_token_pool(self) -> None:
        for ime_type in self.ime_list:
            token_ways = self.ime_handlers[ime_type].tokenize(self.unfreezed_keystrokes)
            for ways in token_ways:
                for token in ways:
                    self._token_pool_set.add((token, self.get_token_distance(token)))

    def _is_token_in_pool(self, token: str) -> bool:
        for token, _ in self.token_pool:
            if token == token:
                return True
        return False

    def get_token_distance(self, request_token: str) -> int:
        for token, distance in self.token_pool:
            if token == request_token:
                return distance

        return self._closest_word_distance(request_token)

    @lru_cache_with_doc(maxsize=128)
    def token_to_candidates(self, token: str) -> list[Candidate]:
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

        candidates = self.token_to_candidates(token)
        return [candidate.word for candidate in candidates]

    def _filter_possible_sentences_by_distance(
        self, possible_sentences: list[list[str]]
    ) -> list[list[str]]:
        result = [
            dict(
                sentence=sentence,
                distance=self._calculate_sentence_distance(sentence),
            )
            for sentence in possible_sentences
        ]
        result = sorted(result, key=lambda x: x["distance"])
        min_distance = result[0]["distance"]
        result = [r for r in result if r["distance"] <= min_distance]
        return result

    def _get_best_sentence(self, possible_sentences: list[dict]) -> list[str]:
        sorted(possible_sentences, key=lambda x: len(x["sentence"]))
        return possible_sentences[0]["sentence"]

    def _token_sentence_to_word_sentence(
        self, token_sentence: list[str], context: str = ""
    ) -> list[str]:

        def solve_sentence(sentence_candidate: list[list[Candidate]], pre_word: str):
            # TODO: Consider the context
            def recursive(best_sentence_tokens: list[list[Candidate]]) -> list[str]:
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
                related_phrases = [
                    phrase
                    for phrase in related_phrases
                    if len(phrase) <= len(best_sentence_tokens)
                ]
                related_phrases = sorted(
                    related_phrases, key=lambda x: len(x), reverse=True
                )

                for phrase in related_phrases:
                    correct_phrase = True
                    for i, char in enumerate(phrase):
                        if char not in [
                            candidate.word for candidate in best_sentence_tokens[i]
                        ]:
                            correct_phrase = False
                            break

                    if correct_phrase:
                        return [c for c in phrase] + recursive(
                            best_sentence_tokens[len(phrase) :]
                        )

                return [c for c in best_sentence_tokens[0][0].word] + recursive(
                    best_sentence_tokens[1:]
                )

            return recursive(sentence_candidate)

        sentence_candidates = [
            self.token_to_candidates(token) for token in token_sentence
        ]

        pre_word = context[-1] if context else ""
        result = solve_sentence(sentence_candidates, pre_word)
        return result

    def _reconstruct_sentence(self, keystroke: str) -> list[list[str]]:
        """
        Reconstruct the sentence back to the keystroke by searching all the
        possible combination of tokens in the token pool.

        Args:
            keystroke (str): The keystroke to search for
        Returns:
            list: A list of **list of str** containing the possible sentences constructed from the token pool
        """

        def dp_search(keystroke: str, token_pool: set[str]) -> list[list[str]]:
            if not keystroke:
                return [[]]

            ans = []
            for token_str in token_pool:
                if keystroke.startswith(token_str):
                    ans.extend(
                        [
                            [token_str] + sub_ans
                            for sub_ans in dp_search(
                                keystroke[len(token_str) :], token_pool
                            )
                        ]
                    )

            if keystroke in token_pool:
                ans.append([keystroke])
            return ans

        token_pool = set(
            [token for token, dis in self.token_pool if dis != float("inf")]
        )
        result = dp_search(keystroke, token_pool)
        unique_result = list(
            map(list, set(map(tuple, result)))
        )  # FIXME: Find out why there are duplicates
        if not unique_result:
            token_pool = set([token for token, dis in self.token_pool])
            result = dp_search(keystroke, token_pool)
            unique_result = list(
                map(list, set(map(tuple, result)))
            )  # FIXME: Find out why there are duplicates

        return unique_result

    def _calculate_sentence_distance(self, sentence: list[str]) -> int:
        """
        Calculate the distance of the sentence based on the token pool.

        Args:
            sentence (list): The sentence to calculate the distance
        Returns:
            int: The distance of the sentence
        """

        return sum([self.get_token_distance(token) for token in sentence])

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

        if not self._is_token_in_pool(token):
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


import keyboard


def get_composition_string_with_cusor(
    total_composition_words: list[str],
    freezed_index: int,
    unfreezed_composition_words: list[int],
    composition_index: int,
) -> str:
    total = []
    for i, word in enumerate(total_composition_words):
        if i < freezed_index:
            total.append((Fore.BLUE if WITH_COLOR else "") + word + Style.RESET_ALL)
        elif freezed_index <= i < freezed_index + len(unfreezed_composition_words):
            total.append((Fore.YELLOW if WITH_COLOR else "") + word + Style.RESET_ALL)
        else:
            total.append((Fore.BLUE if WITH_COLOR else "") + word + Style.RESET_ALL)

    total.insert(composition_index, Fore.GREEN + "|" + Style.RESET_ALL)
    return "".join(total)


def get_candidate_words_with_cursor(
    candidate_word_list: list[str], selection_index: int
) -> str:
    output = "["
    for i, word in enumerate(candidate_word_list):
        if i == selection_index:
            output += Fore.GREEN + word + Style.RESET_ALL + " "
        else:
            output += word + " "
    output += "]"
    return output


class EventWrapper:
    def __init__(self):
        start_time = time.time()
        self.my_keyeventhandler = KeyEventHandler(verbose_mode=False)
        print("Initialization time: ", time.time() - start_time)
        self.last_key_event = None
        self.timer = None

    def on_press_handler(self, event):
        if event.name in ["enter", "left", "right", "down", "up", "esc"]:
            self.my_keyeventhandler.handle_key(event.name)
            self.update_ui()
        else:
            if self.timer is not None:
                self.timer.cancel()

            total_composition_words = None
            freezed_index = self.my_keyeventhandler.freezed_index
            freezed_composition_words = (
                self.my_keyeventhandler.freezed_composition_words
            )
            unfreezed_composition_words = (
                self.my_keyeventhandler.unfreezed_composition_words
            )
            composition_index = self.my_keyeventhandler.composition_index

            in_selection_mode = self.my_keyeventhandler.in_selection_mode
            selection_index = self.my_keyeventhandler.selection_index
            candidate_word_list = self.my_keyeventhandler.candidate_word_list

            # quick action
            if event.name == "backspace":
                unfreezed_composition_words = unfreezed_composition_words[:-1]  # ! ??
                total_composition_words = (
                    freezed_composition_words[:freezed_index]
                    + unfreezed_composition_words
                    + freezed_composition_words[freezed_index:]
                )
                composition_index -= 1
            elif event.name == "space":
                unfreezed_composition_words = unfreezed_composition_words + [" "]
                total_composition_words = (
                    freezed_composition_words[:freezed_index]
                    + unfreezed_composition_words
                    + freezed_composition_words[freezed_index:]
                )
                composition_index += 1
            elif event.name in TOTAL_VALID_KEYSTROKE_SET:
                unfreezed_composition_words = unfreezed_composition_words + [event.name]
                total_composition_words = (
                    freezed_composition_words[:freezed_index]
                    + unfreezed_composition_words
                    + freezed_composition_words[freezed_index:]
                )
                composition_index += 1

            else:
                print(f"Invalid key: {event.name}")
                return

            self.my_keyeventhandler.handle_key(event.name)

            self.timer = threading.Timer(0.5, self.update_ui)
            self.timer.start()

            print(
                f"{get_composition_string_with_cusor(total_composition_words, freezed_index, unfreezed_composition_words, composition_index)}"
                + f"\t\t {composition_index}"
                + f"\t\t{get_candidate_words_with_cursor(candidate_word_list, selection_index) if in_selection_mode else ''}"
                + f"\t\t{selection_index if in_selection_mode else ''}"
            )

    def update_ui(self):
        # Slow Action
        print(
            f"{get_composition_string_with_cusor(self.my_keyeventhandler.total_composition_words, self.my_keyeventhandler.freezed_index, self.my_keyeventhandler.unfreezed_composition_words, self.my_keyeventhandler.composition_index)}"
            + f"\t\t {self.my_keyeventhandler.composition_index}"
            + f"\t\t{get_candidate_words_with_cursor(self.my_keyeventhandler.candidate_word_list, self.my_keyeventhandler.selection_index) if self.my_keyeventhandler.in_selection_mode else ''}"
            + f"\t\t{self.my_keyeventhandler.selection_index if self.my_keyeventhandler.in_selection_mode else ''}"
        )

    def run(self):
        keyboard.on_press(self.on_press_handler)
        keyboard.wait("esc")


if __name__ == "__main__":
    event_wrapper = EventWrapper()
    event_wrapper.run()
