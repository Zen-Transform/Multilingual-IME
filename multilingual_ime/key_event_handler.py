import sys
import logging

from pathlib import Path

from .candidate import Candidate
from .keystroke_map_db import KeystrokeMappingDB
from .core.custom_decorators import lru_cache_with_doc
from .core.F import is_chinese_character
from .ime import (
    IMEFactory,
    ENGLISH_IME,
    BOPOMOFO_VALID_KEYSTROKE_SET,
    BOPOMOFO_VALID_SPECIAL_KEYSTROKE_SET,
    ENGLISH_VALID_KEYSTROKE_SET,
    ENGLISH_VALID_SPECIAL_KEYSTROKE_SET,
    PINYIN_VALID_KEYSTROKE_SET,
    CANGJIE_VALID_KEYSTROKE_SET,
    JAPANESE_VALID_KEYSTROKE_SET,
    separate_by_control_characters,
)
from .phrase_db import PhraseDataBase
from .multi_config import MultiConfig
from .sentence_graph import SentenceGraph

FUNCTIONAL_KEYS = [
    "up",
    "down",
    "left",
    "right",
    "enter",
    "backspace",
    "delete",
    "tab",
    "escape",
    "CapsLock",
]

TOTAL_VALID_KEYSTROKE_SET = (
    BOPOMOFO_VALID_KEYSTROKE_SET.union(ENGLISH_VALID_KEYSTROKE_SET)
    .union(PINYIN_VALID_KEYSTROKE_SET)
    .union(CANGJIE_VALID_KEYSTROKE_SET)
    .union(JAPANESE_VALID_KEYSTROKE_SET)
    .union(BOPOMOFO_VALID_SPECIAL_KEYSTROKE_SET)
    .union(ENGLISH_VALID_SPECIAL_KEYSTROKE_SET)
)

CHINESE_PHRASE_DB_PATH = Path(__file__).parent / "src" / "chinese_phrase.db"
USER_PHRASE_DB_PATH = Path(__file__).parent / "src" / "user_phrase.db"
USER_FREQUENCY_DB_PATH = Path(__file__).parent / "src" / "user_frequency.db"

MAX_SAVE_PRE_POSSIBLE_SENTENCES = 5


class KeyEventHandler:
    """ 
    The **core** of the Multilingual IME.
    Handles key events and manages the state of the IME.
    This class is responsible for processing key events, managing the composition string,
    and interacting with the phrase and frequency databases.

    Note:
        - This class is designed to be used as a singleton, \
        and should not be instantiated multiple times.
        It is recommended to use the `KeyEventHandler()` directly in the main application.    
        - A `token` is a string of keystrokes that represents a word in the IMEs. \
          A `token` can convert to various words in different IMEs.
        - A `word` is a string of characters that represents a valid word in the IMEs. \
          A `word` can be a single character (Chinese character) or a string of characters \
          (English word).
        - A `phrase` is a string of word (Chinese character) that represents a valid phrase.
    """

    def __init__(self, verbose_mode: bool = False) -> None:
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose_mode else logging.WARNING)
        self.logger.addHandler(logging.StreamHandler())

        # Setup Config
        self._config = MultiConfig()
        self._chinese_phrase_db = PhraseDataBase(CHINESE_PHRASE_DB_PATH)
        self._user_phrase_db = PhraseDataBase(USER_PHRASE_DB_PATH)
        self._user_frequency_db = KeystrokeMappingDB(USER_FREQUENCY_DB_PATH)

        # Setup IMEs
        self.ime_handlers = {
            ime: IMEFactory.create_ime(ime) for ime in self.activated_imes
        }

        # Config Settings
        self.auto_phrase_learn = self._config.AUTO_PHRASE_LEARN
        self.auto_frequency_learn = self._config.AUTO_FREQUENCY_LEARN
        self.selection_page_size = self._config.SELECTION_PAGE_SIZE

        self.commit_string = ""
        # State Variables
        self._freezed_index = 0
        self.unfreeze_keystrokes = ""
        self._freezed_candidate_sentence: list[Candidate] = []
        self._unfreeze_candidate_sentence: list[Candidate] = []

        # Selection States
        self.in_selection_mode = False
        self._total_selection_index = 0
        self._total_candidate_list: list[Candidate] = []

    def _reset_composition_states(self) -> None:
        self._freezed_index = 0
        self.unfreeze_keystrokes = ""
        self._freezed_candidate_sentence = []
        self._unfreeze_candidate_sentence = []

        self._reset_selection_states()

    def _reset_selection_states(self) -> None:
        self.in_selection_mode = False
        self._total_selection_index = 0
        self._total_candidate_list = []

    def _unfreeze_to_freeze(self) -> None:
        additional_candidate_sentence = self._separate_english_candidate(
            self._unfreeze_candidate_sentence
        )
        self._freezed_candidate_sentence = (
            self._freezed_candidate_sentence[: self._freezed_index]
            + additional_candidate_sentence
            + self._freezed_candidate_sentence[self._freezed_index :]
        )
        self._unfreeze_candidate_sentence = []
        self.unfreeze_keystrokes = ""
        self._freezed_index += len(additional_candidate_sentence)

    def _separate_english_candidate(
        self, candidate_sentence: list[Candidate]
    ) -> list[Candidate]:
        return_sentence = []
        for candidate in candidate_sentence:
            if candidate.ime_method == ENGLISH_IME:
                # Separate the english word by character
                return_sentence.extend(
                    [
                        Candidate(
                            c,
                            c,
                            0,
                            c,
                            0,
                            ENGLISH_IME,
                        )
                        for c in candidate.word
                    ]
                )
            else:
                return_sentence.append(candidate)
        return return_sentence

    def set_activation_status(self, ime_type: str, status: bool) -> None:
        """
        Set the activation status of the IME.
        Args:
            ime_type (str): The type of the IME to set the status for.
            status (bool): The activation status to set.
        Raises:
            ValueError: If the IME type is not found in the config.
            ValueError: If the status is not a boolean value.
        """
        # FIXME: expose config directly/ fix raise ValueError
        self._config.setIMEActivationStatus(ime_name=ime_type, status=status)

    @property
    def activated_imes(self) -> list[str]:
        """
        A list of activated IMEs based on the config.
        """
        return self._config.ACTIVE_IME

    @property
    def _total_candidate_sentence(self) -> list[Candidate]:
        """
        The total candidate sentence as a list of Candidate objects.
        This includes both freezed and unfreeze candidate sentences.
        """
        return (
            self._freezed_candidate_sentence[: self._freezed_index]
            + self._unfreeze_candidate_sentence
            + self._freezed_candidate_sentence[self._freezed_index :]
        )

    @property
    def composition_index(self) -> int:
        """
        The current cursor index in the composition string.

        """
        return self._freezed_index + self.unfreeze_index

    @property
    def unfreeze_index(self) -> int:
        """
        The index of the current unfreeze composition word.
        """
        return len(self._unfreeze_candidate_sentence)

    @property
    def candidate_word_list(self) -> list[str]:
        """
        The candidate word list for the current token in selection mode.
        Show only the current page of the candidate word list.
        """
        page = self._total_selection_index // self.selection_page_size
        words = [c.word for c in self._total_candidate_list]
        return words[
            page * self.selection_page_size : (page + 1) * self.selection_page_size
        ]

    @property
    def selection_index(self) -> int:
        """
        The index of the current selection in the candidate word list.
        """
        return self._total_selection_index % self.selection_page_size

    @property
    def total_composition_words(self) -> list[str]:
        """
        The total composition words as a list of strings.
        This includes both freezed and unfreeze candidate sentences.
        """
        return [c.word for c in self._total_candidate_sentence]

    @property
    def composition_string(self) -> str:
        """
        The current composition string, which is the concatenation of all composition words.
        """
        return "".join([c.word for c in self._total_candidate_sentence])

    @property
    def in_typing_mode(self) -> bool:
        """
        Check if the IME is in typing mode.
        Typing mode is when the composition string is not empty
        """
        return bool(self.composition_string)

    def handle_key(self, key: str) -> bool:
        """
        Handle the key event and update the composition string and selection states.
        Args:
            key (str): The key event to handle.
        Returns:
            bool: True if the key is handled, False otherwise.
        """
        # Reset the commit string if it is not empty
        if self.commit_string:
            self.logger.info(
                "Commit string: (%s) is not empty, reset to empty", self.commit_string
            )
            self.commit_string = ""

        # Check if the key is valid (key filtering)
        if key in TOTAL_VALID_KEYSTROKE_SET:
            self.handle_normal_key(key)
            return True
        if self.in_typing_mode and key in FUNCTIONAL_KEYS:
            self.handle_functional_key(key)
            return True

        self.logger.info("Unhandled key (pass to OS): %s", key)
        return False

    def handle_functional_key(self, key: str) -> None:
        """
        Handle the functional key events.
        Args:
            key (str): The functional key event to handle.
        Raises:
            ValueError: If the key is not a valid functional key.
        """
        if key not in FUNCTIONAL_KEYS:
            raise ValueError(f"Invalid functional key: {key}")

        if self.in_selection_mode:
            if key == "down":
                self._total_selection_index = (self._total_selection_index + 1) % len(
                    self._total_candidate_list
                )
            elif key == "up":
                self._total_selection_index = (self._total_selection_index - 1) % len(
                    self._total_candidate_list
                )
            elif (
                key == "enter"
            ):  # Overwrite the composition string & reset selection states
                selected_candidate = self._total_candidate_list[
                    self._total_selection_index
                ]
                self._freezed_candidate_sentence[self.composition_index - 1] = (
                    selected_candidate
                )

                self._reset_selection_states()
            elif key == "left":  # Open side selection ?
                pass
            elif key == "right":
                pass
            elif key == "esc":
                self._reset_selection_states()
            else:
                self.logger.info(
                    "Unhandled functional key (in selection mode): %s", key
                )

            return
        else:
            if (
                key == "enter"
            ):  # Commit the composition string, update the db & reset all states
                self.commit_string = self.composition_string
                self._unfreeze_to_freeze()
                if self.auto_phrase_learn:
                    self.update_user_phrase_db(self.composition_string)
                if self.auto_frequency_learn:
                    self.update_user_frequency_db()
                self._reset_composition_states()
            elif key == "left":
                self._unfreeze_to_freeze()
                if self._freezed_index > 0:
                    self._freezed_index -= 1
            elif key == "right":
                self._unfreeze_to_freeze()
                if self._freezed_index < len(self._total_candidate_sentence):
                    self._freezed_index += 1
            elif key == "down":  # Enter selection mode
                self._unfreeze_to_freeze()
                if (
                    len(self._total_candidate_sentence) > 0
                    and self.composition_index > 0
                ):
                    candidate = self._total_candidate_sentence[
                        self.composition_index - 1
                    ]
                    if candidate.ime_method != ENGLISH_IME:
                        token = candidate.keystrokes
                        self._total_candidate_list = self.token_to_candidates(token)
                        if len(self._total_candidate_list) > 1:
                            # Only none-english token can enter selection mode, and
                            # the candidate list should have more than 1 candidate
                            self.in_selection_mode = True
                            self._total_selection_index = 0
            elif key == "esc":
                self._reset_composition_states()
            elif key == "backspace":
                if self.unfreeze_index > 0:
                    self.unfreeze_keystrokes = self.unfreeze_keystrokes[:-1].rstrip("©")
                    last_candidate_keystroke = self._unfreeze_candidate_sentence[
                        -1
                    ].keystrokes
                    self._unfreeze_candidate_sentence = (
                        self._unfreeze_candidate_sentence[:-1]
                        + [
                            Candidate(
                                last_candidate_keystroke[:-1],
                                last_candidate_keystroke[:-1],
                                0,
                                last_candidate_keystroke[:-1],
                                0,
                                ENGLISH_IME,
                            )
                        ]
                    )
                else:
                    if self._freezed_index > 0:
                        self._freezed_candidate_sentence = (
                            self._freezed_candidate_sentence[: self._freezed_index - 1]
                            + self._freezed_candidate_sentence[self._freezed_index :]
                        )
                        self._freezed_index -= 1
                        return
            elif key == "delete":
                if self._freezed_index >= 0 and self._freezed_index < len(
                    self._freezed_candidate_sentence
                ):
                    self._freezed_candidate_sentence = (
                        self._freezed_candidate_sentence[: self._freezed_index]
                        + self._freezed_candidate_sentence[self._freezed_index + 1 :]
                    )
            else:
                self.logger.info("Unhandled functional key: %s", key)

            return

    def handle_normal_key(self, key: str) -> None:
        """
        Handle the normal key events.
        Args:
            key (str): The normal key event to handle.
        Raises:
            ValueError: If the key is not a valid normal key.
        """
        if key not in TOTAL_VALID_KEYSTROKE_SET:
            raise ValueError(f"Invalid normal key: {key}")

        if (
            self.in_selection_mode
        ):  # If in selection mode and keep typing, reset the selection states
            self._reset_selection_states()

        if key in TOTAL_VALID_KEYSTROKE_SET:
            self.unfreeze_keystrokes += key
        elif key.startswith("©"):
            self.unfreeze_keystrokes += key
        else:
            self.logger.info("Unhandled normal key: %s", key)
            return

    def slow_handle(self):
        """
        Handle the slow process of constructing the unfreeze token \
        sentence from the unfreeze keystrokes.
        """
        token_sentences = self._separate_tokens(self.unfreeze_keystrokes)
        self._unfreeze_candidate_sentence = self._token_sentence_to_candidate_sentence(
            token_sentences
        )

    @lru_cache_with_doc(maxsize=128)
    def get_token_distance(self, token: str) -> int:
        """
        Get the distance of the given token to its closest word from all IMEs

        Args:
            token (str): The token to search for
        Returns:
            int: The distance to the closest word
        """
        min_distance = sys.maxsize

        for ime_type in self.activated_imes:
            if not self.ime_handlers[ime_type].is_valid_token(token):
                continue

            method_distance = self.ime_handlers[ime_type].closest_word_distance(token)
            min_distance = min(min_distance, method_distance)
            if min_distance == 0:
                break
        return min_distance

    def token_to_candidates(self, token: str) -> list[Candidate]:
        """
        Get the possible candidates of the token from all IMEs.

        Args:
            token (str): The token to search for
        Returns:
            list: A list of **Candidate** containing the possible candidates
        """
        candidates = []

        for ime_type in self.activated_imes:
            if self.ime_handlers[ime_type].is_valid_token(token):
                candidates.extend(
                    self.ime_handlers[ime_type].get_token_candidates(token)
                )

        if len(candidates) == 0:
            self.logger.info("No candidates found for token '%s'", token)
            # If no candidates found, treat the token as a single candidate of English IME
            return [Candidate(token, token, 0, token, 0, ENGLISH_IME)]

        # First sort by distance
        candidates = sorted(candidates, key=lambda x: x.distance)

        # Filter out the candidates with distance > smallest_distance
        smallest_distance = candidates[0].distance
        candidates = filter(lambda x: x.distance <= smallest_distance, candidates)

        # Then sort by frequency
        candidates = sorted(candidates, key=lambda x: x.word_frequency, reverse=True)

        # This is a hack to increase the rank of the token if it is in the user frequency db
        new_candidates = []
        for candidate in candidates:
            if self._user_frequency_db.word_exists(candidate.word):
                new_candidates.append(
                    (
                        candidate,
                        self._user_frequency_db.get_word_frequency(candidate.word),
                    )
                )
            else:
                new_candidates.append((candidate, 0))
        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        candidates = [candidate[0] for candidate in new_candidates]

        return candidates

    def _token_sentence_to_candidate_sentence(
        self, token_sentence: list[str], context: str = "", naive_first: bool = False
    ) -> list[Candidate]:

        def solve_sentence_phrase_matching(
            sentence_candidate: list[list[Candidate]], pre_word: str = ""
        ) -> list[Candidate]:
            # TODO: Consider the context
            def recursive(
                best_sentence_tokens: list[list[Candidate]],
            ) -> list[Candidate]:
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
                related_phrases = sorted(related_phrases, key=len, reverse=True)

                for phrase in related_phrases:
                    if all(
                        phrase[i] in [c.word for c in best_sentence_tokens[i]]
                        for i in range(len(phrase))
                    ):
                        phrase_candidates = [
                            next(
                                c
                                for c in best_sentence_tokens[i]
                                if c.word == phrase[i]
                            )
                            for i in range(len(phrase))
                        ]
                        return phrase_candidates + recursive(
                            best_sentence_tokens[len(phrase) :]
                        )

                return [best_sentence_tokens[0][0]] + recursive(
                    best_sentence_tokens[1:]
                )

            return recursive(sentence_candidate)

        def solve_sentence_naive_first(
            sentence_candidate: list[list[Candidate]],
        ) -> list[Candidate]:
            return [c[0] for c in sentence_candidate]

        sentence_candidates = [
            self.token_to_candidates(token) for token in token_sentence
        ]

        if naive_first:
            return solve_sentence_naive_first(sentence_candidates)

        pre_word = context[-1] if context else ""
        result = solve_sentence_phrase_matching(sentence_candidates, pre_word)
        return result

    def _calculate_sentence_distance(self, sentence: list[str]) -> int:
        """
        Calculate the distance of the sentence based on the token pool.

        Args:
            sentence (list): The sentence to calculate the distance
        Returns:
            int: The distance of the sentence
        """
        return sum([self.get_token_distance(token) for token in sentence])

    def update_user_frequency_db(self) -> None:
        for candidate in self._total_candidate_sentence:
            if len(candidate.word) == 1 and is_chinese_character(candidate.word):
                if not self._user_frequency_db.word_exists(candidate.word):
                    self._user_frequency_db.insert(
                        candidate.keystrokes, candidate.word, 1
                    )
                else:
                    self._user_frequency_db.increment_word_frequency(candidate.word)

    def update_user_phrase_db(self, text: str) -> None:
        raise NotImplementedError("update_user_phrase_db is not implemented yet")

    def _separate_tokens(self, keystroke: str) -> list[str]:
        """
        The function to separate the keystrokes into tokens.
        Separate the keystrokes into tokens by most logical way.

        Args:
            keystroke (str): The keystrokes to separate into tokens.
            top_n (int): The maximum number of possible separation results to return.
        Returns:
            list[list[str]]: A list of possible token sentences, each sentence is a list of tokens.
        """
        # In the older version, the function is named as `new_reconstruct()`

        if not keystroke:
            return []

        # Get all possible seps
        possible_seps = []
        for ime_type in self.activated_imes:
            token_ways = self.ime_handlers[ime_type].tokenize(keystroke)
            possible_seps.extend(token_ways)

        # sep by ©
        possible_seps.append(
            separate_by_control_characters(keystroke, control_char="©")
        )
        # Filter out empty sep
        possible_seps = [sep for sep in possible_seps if sep]
        # Filter out same sep
        possible_seps = [list(t) for t in set(tuple(token) for token in possible_seps)]

        token_pool = {token for sep in possible_seps for token in sep}
        new_possible_seps = []
        for sep_tokens in possible_seps:
            new_sep = []
            for token in sep_tokens:
                is_sep = False
                for i in range(1, len(token)):
                    if token[:i] in token_pool:
                        new_sep.extend([token[:i], token[i:]])
                        is_sep = True
                        break
                if not is_sep:
                    new_sep.append(token)

            new_possible_seps.append(new_sep)
        new_possible_seps.extend(possible_seps)

        # Filter out new same sep
        new_possible_seps = [
            list(t) for t in set(tuple(token) for token in possible_seps)
        ]
        self.logger.info("New Possible seps: %s", new_possible_seps)
        self.logger.info("Creating Graph with %d possible seps", len(new_possible_seps))

        sentence_graph = SentenceGraph()
        for sep_tokens in new_possible_seps:
            sep_tokens = [
                (token, self.get_token_distance(token)) for token in sep_tokens
            ]
            self.logger.info("Adding token path: %s", sep_tokens)
            sentence_graph.add_token_path(sep_tokens)

        possible_paths = sentence_graph.get_sentence()

        self.logger.info(
            "Found %d possible paths %s",
            len(possible_paths),
            possible_paths,
        )
        return possible_paths[0]

    def end_to_end(self, keystroke: str) -> list[str]:
        """
        End-to-end reconstruction of the keystroke to a sentence.
        Args:
            keystroke (str): The keystrokes to convert to a sentence.
        Returns:
            list[str]: A list of words reconstructed from the keystrokes.
        """
        token_sentences = self._separate_tokens(keystroke)
        if not token_sentences:
            return []
        candidate_sentences = self._token_sentence_to_candidate_sentence(
            token_sentences
        )
        return [candidate.word for candidate in candidate_sentences]


if __name__ == "__main__":
    handler = KeyEventHandler()
    handler.set_activation_status("japanese", True)
    handler.set_activation_status("bopomofo", True)
    handler.set_activation_status("cangjie", False)
    handler.set_activation_status("english", True)
    handler.set_activation_status("pinyin", False)
    test_case = "hello world"
    new_result = handler._separate_tokens(test_case)
    print("---------------------")
    print("NEW", new_result)
    print("---------------------")
    print("NEW", handler._calculate_sentence_distance(new_result))
    print("---------------------")
    print("PHASE1", handler.end_to_end(test_case))
    print("NEW", handler.end_to_end(test_case))
    print("---------------------")
