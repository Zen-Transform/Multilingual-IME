import logging

from pathlib import Path

from .candidate import Candidate
from .keystroke_map_db import KeystrokeMappingDB
from .core.custom_decorators import lru_cache_with_doc
from .core.F import (
    modified_levenshtein_distance,
    is_chinese_character,
)
from .ime import (
    IMEFactory,
    ENGLISH_IME,
)
from .phrase_db import PhraseDataBase
from .muti_config import MultiConfig
from .sentence_graph import SentenceGraph

from .ime import (
    BOPOMOFO_VALID_KEYSTROKE_SET,
    ENGLISH_VALID_KEYSTROKE_SET,
    PINYIN_VALID_KEYSTROKE_SET,
    CANGJIE_VALID_KEYSTROKE_SET,
    JAPANESE_VALID_KEYSTROKE_SET,
)

TOTAL_VALID_KEYSTROKE_SET = (
    BOPOMOFO_VALID_KEYSTROKE_SET.union(ENGLISH_VALID_KEYSTROKE_SET)
    .union(PINYIN_VALID_KEYSTROKE_SET)
    .union(CANGJIE_VALID_KEYSTROKE_SET)
    .union(JAPANESE_VALID_KEYSTROKE_SET)
)

CHINESE_PHRASE_DB_PATH = Path(__file__).parent / "src" / "chinese_phrase.db"
USER_PHRASE_DB_PATH = Path(__file__).parent / "src" / "user_phrase.db"
USER_FREQUENCY_DB_PATH = Path(__file__).parent / "src" / "user_frequency.db"

MAX_SAVE_PRE_POSSIBLE_SENTENCES = 5


class KeyEventHandler:
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

        # State Variables
        self._token_pool_set = set()
        self._pre_possible_sentences = []
        self.have_selected = False

        self.freezed_index = 0
        self.freezed_token_sentence = []
        self.freezed_composition_words = []

        self.unfreeze_keystrokes = ""
        self.unfreeze_token_sentence = []
        self.unfreeze_composition_words = []
        self.commit_string = ""

        # Selection States
        self.in_selection_mode = False
        self._total_selection_index = 0
        self._total_candidate_word_list = []

    def _reset_all_states(self) -> None:
        self._token_pool_set = set()
        self._pre_possible_sentences = []
        self.have_selected = False

        self.freezed_index = 0
        self.freezed_token_sentence = []
        self.freezed_composition_words = []

        self.unfreeze_keystrokes = ""
        self.unfreeze_token_sentence = []
        self.unfreeze_composition_words = []

        self._reset_selection_states()

    def _reset_selection_states(self) -> None:
        self.in_selection_mode = False
        self._total_selection_index = 0
        self._total_candidate_word_list = []

    def _unfreeze_to_freeze(self) -> None:
        self._token_pool_set = set()
        self._pre_possible_sentences = []
        self.freezed_token_sentence = self.separate_english_token(
            self.total_token_sentence
        )  # Bad design here
        self.freezed_composition_words = self.separate_english_token(
            self.total_composition_words
        )
        self.freezed_index = self.freezed_index + len(
            self.separate_english_token(self.unfreeze_composition_words)
        )

        self.unfreeze_keystrokes = ""
        self.unfreeze_token_sentence = []
        self.unfreeze_composition_words = []

    def separate_english_token(self, tokens: list[str]) -> list[str]:
        #  Special case for English, separate the english word by character
        result = []
        for token in tokens:
            if self.ime_handlers[ENGLISH_IME].is_valid_token(token):
                result.extend([c for c in token])
            else:
                result.append(token)
        return result

    def set_activation_status(self, ime_type: str, status: bool) -> None:
        self._config.setIMEActivationStatus(ime_name=ime_type, status=status)

    @property
    def activated_imes(self) -> list[str]:
        return self._config.ACTIVE_IME

    @property
    def token_pool(self) -> list[str]:
        return list(self._token_pool_set)

    @property
    def total_composition_words(self) -> list[str]:
        return (
            self.freezed_composition_words[: self.freezed_index]
            + self.unfreeze_composition_words
            + self.freezed_composition_words[self.freezed_index :]
        )

    @property
    def total_token_sentence(self) -> list[str]:
        return (
            self.freezed_token_sentence[: self.freezed_index]
            + self.unfreeze_token_sentence
            + self.freezed_token_sentence[self.freezed_index :]
        )

    @property
    def composition_index(self) -> int:
        return self.freezed_index + self.unfreeze_index

    @property
    def unfreeze_index(self) -> int:
        return len(self.unfreeze_composition_words)

    @property
    def candidate_word_list(self) -> list[str]:
        """
        The candidate word list for the current token in selection mode.
        Show only the current page of the candidate word list.
        """
        page = self._total_selection_index // self.selection_page_size
        return self._total_candidate_word_list[
            page * self.selection_page_size : (page + 1) * self.selection_page_size
        ]

    @property
    def selection_index(self) -> int:
        return self._total_selection_index % self.selection_page_size

    @property
    def composition_string(self) -> str:
        return "".join(self.total_composition_words)

    def handle_key(self, key: str) -> None:
        special_keys = ["enter", "left", "right", "down", "up", "esc"]
        if self.commit_string:
            self.logger.info(
                "Commit string: (%s) is not empty, reset to empty", self.commit_string
            )
            self.commit_string = ""

        if key in special_keys:
            if self.in_selection_mode:
                if key == "down":
                    if (
                        self._total_selection_index
                        < len(self._total_candidate_word_list) - 1
                    ):
                        self._total_selection_index += 1
                elif key == "up":
                    if self._total_selection_index > 0:
                        self._total_selection_index -= 1
                elif (
                    key == "enter"
                ):  # Overwrite the composition string & reset selection states
                    self.have_selected = True
                    selected_word = self._total_candidate_word_list[
                        self._total_selection_index
                    ]
                    self.freezed_composition_words[self.composition_index - 1] = (
                        selected_word
                    )
                    # ! Recalculate the index
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
                if (
                    key == "enter"
                ):  # Commit the composition string, update the db & reset all states
                    self.commit_string = self.composition_string
                    self._unfreeze_to_freeze()
                    if self.auto_phrase_learn:
                        self.update_user_phrase_db(self.composition_string)
                    if self.auto_frequency_learn:
                        self.update_user_frequency_db()
                    self._reset_all_states()
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
                        token = self.total_token_sentence[self.composition_index - 1]
                        if not self.ime_handlers[ENGLISH_IME].is_valid_token(token):
                            self._total_candidate_word_list = (
                                self._get_token_candidate_words(token)
                            )
                            if len(self._total_candidate_word_list) > 1:
                                # Only none-english token can enter selection mode, and
                                # the candidate list should have more than 1 candidate
                                self.in_selection_mode = True
                elif key == "esc":
                    self._reset_all_states()
                else:
                    print(f"Invalid Special key: {key}")

                return
        else:
            if (
                self.in_selection_mode
            ):  # If in selection mode and keep typing, reset the selection states
                self._reset_selection_states()

            if key == "backspace":
                if self.unfreeze_index > 0:
                    self.unfreeze_keystrokes = self.unfreeze_keystrokes[:-1]
                    self.unfreeze_composition_words = self.unfreeze_composition_words[
                        :-1
                    ] + [self.unfreeze_token_sentence[-1][:-1]]
                else:
                    if self.freezed_index > 0:
                        self.freezed_composition_words = (
                            self.freezed_composition_words[: self.freezed_index - 1]
                            + self.freezed_composition_words[self.freezed_index :]
                        )
                        self.freezed_index -= 1
                        return
            elif key == "space":
                self.unfreeze_keystrokes += " "
                self.unfreeze_composition_words += [" "]
            elif key in TOTAL_VALID_KEYSTROKE_SET:
                self.unfreeze_keystrokes += key
                self.unfreeze_composition_words += [key]
            elif key.startswith("©"):
                self.unfreeze_keystrokes += key
                self.unfreeze_composition_words += [key[1:]]
            else:
                print(f"Invalid key: {key}")
                return

    def slow_handle(self):
        # This is the V2 of the handle_key function, using the new_reconstruct function
        token_sentences = self.new_reconstruct(self.unfreeze_keystrokes)
        if not token_sentences:
            return
        self.unfreeze_token_sentence = token_sentences[0]
        self.unfreeze_composition_words = self._token_sentence_to_word_sentence(
            self.unfreeze_token_sentence
        )

    def _update_token_pool(self) -> None:
        for ime_type in self.activated_imes:
            token_ways = self.ime_handlers[ime_type].tokenize(self.unfreeze_keystrokes)
            for ways in token_ways:
                for token in ways:
                    self._token_pool_set.add(token)

        # Cut large token to small token
        # TODO: This is a hack, need to find a better way to handle this
        sorted_tokens = sorted(list(self._token_pool_set), key=len, reverse=True)
        for token in sorted_tokens:
            if len(token) > 1:
                for i in range(1, len(token)):
                    if token[:i] in self._token_pool_set:
                        self._token_pool_set.add(token[i:])

    def _is_token_in_pool(self, token: str) -> bool:
        return token in self._token_pool_set

    @lru_cache_with_doc(maxsize=128)
    def get_token_distance(self, token: str) -> int:
        """
        Get the distance of the given token to its closest word from all IMEs

        Args:
            token (str): The token to search for
        Returns:
            int: The distance to the closest word
        """
        min_distance = float("inf")

        for ime_type in self.activated_imes:
            if not self.ime_handlers[ime_type].is_valid_token(token):
                continue

            method_distance = self.ime_handlers[ime_type].closest_word_distance(token)
            min_distance = min(min_distance, method_distance)
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
            self.logger.info("No candidates found for token '%s'", token)
            return [Candidate(token, token, 0, token, 0, "NO_IME")]

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

    def _get_token_candidate_words(self, token: str) -> list[str]:
        """
        Get the possible candidate words of the token from all IMEs.

        Args:
            token (str): The token to search for
        Returns:
            list: A list of **str** containing the possible candidate words
        """

        candidates = self.token_to_candidates(token)
        return [candidate.word for candidate in candidates]

    def _sort_possible_sentences(
        self, possible_sentences: list[list[str]]
    ) -> list[list[str]]:
        # Sort the possible sentences by the distance
        possible_sentences_with_distance = [
            {
                "sentence": sentence,
                "distance": self._calculate_sentence_distance(sentence),
            }
            for sentence in possible_sentences
        ]
        possible_sentences_with_distance = sorted(
            possible_sentences_with_distance, key=lambda x: x["distance"]
        )
        min_distance = possible_sentences_with_distance[0]["distance"]
        possible_sentences_with_distance = [
            r for r in possible_sentences_with_distance if r["distance"] <= min_distance
        ]

        # Sort the possible sentences by the number of tokens
        possible_sentences = sorted(
            possible_sentences_with_distance, key=lambda x: len(x["sentence"])
        )
        return [r["sentence"] for r in possible_sentences]

    def _token_sentence_to_word_sentence(
        self, token_sentence: list[str], context: str = "", naive_first: bool = False
    ) -> list[str]:

        def solve_sentence_phrase_matching(
            sentence_candidate: list[list[Candidate]], pre_word: str = ""
        ):
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
                related_phrases = sorted(related_phrases, key=len, reverse=True)

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

                return [best_sentence_tokens[0][0].word] + recursive(
                    best_sentence_tokens[1:]
                )

            return recursive(sentence_candidate)

        def solve_sentence_naive_first(
            sentence_candidate: list[list[Candidate]],
        ) -> list[str]:
            return [c[0].word for c in sentence_candidate]

        sentence_candidates = [
            self.token_to_candidates(token) for token in token_sentence
        ]

        if naive_first:
            return solve_sentence_naive_first(sentence_candidates)

        pre_word = context[-1] if context else ""
        result = solve_sentence_phrase_matching(sentence_candidates, pre_word)
        return result

    def _reconstruct_sentence_from_pre_possible_sentences(
        self, target_keystroke: str
    ) -> list[list[str]]:
        try:
            possible_sentences = []

            if self._pre_possible_sentences != []:
                current_best_sentence = "".join(self._pre_possible_sentences[0])

                if len(target_keystroke) >= len(current_best_sentence):
                    for pre_possible_sentence in self._pre_possible_sentences:
                        subtracted_string = target_keystroke[
                            len("".join(pre_possible_sentence[:-1])) :
                        ]  # Get the remaining string that haven't been processed
                        possible_sentences.extend(
                            [
                                pre_possible_sentence[:-1] + sub_sentence_results
                                for sub_sentence_results in self._reconstruct_sentence(
                                    subtracted_string
                                )
                            ]
                        )
                else:  # The target_keystroke is shorter than the current best sentence, (e.g. backspace)
                    for pre_possible_sentence in self._pre_possible_sentences:
                        if "".join(pre_possible_sentence[:-1]).startswith(
                            target_keystroke
                        ):
                            possible_sentences.append(pre_possible_sentence[:-1])

                assert (
                    possible_sentences != []
                ), "No possible sentences found in the case of pre_possible_sentences"
            else:
                possible_sentences = self._reconstruct_sentence(target_keystroke)
        except AssertionError as e:
            self.logger.info(e)
            possible_sentences = self._reconstruct_sentence(target_keystroke)

        return possible_sentences

    def _reconstruct_sentence(self, keystroke: str) -> list[list[str]]:
        """
        Reconstruct the sentence back to the keystroke by searching all the
        possible combination of tokens in the token pool.

        Args:
            keystroke (str): The keystroke to search for
        Returns:
            list: A list of **list of str** containing the \
            possible sentences constructed from the token pool
        """

        def dp_search(keystroke: str, token_pool: set[str]) -> list[list[str]]:
            if not keystroke:
                return []

            ans = []
            for token_str in token_pool:
                if keystroke.startswith(token_str):
                    ans.extend(
                        [
                            [token_str] + sub_ans
                            for sub_ans in dp_search(
                                keystroke[len(token_str) :], token_pool
                            )
                            if sub_ans
                        ]
                    )

            if keystroke in token_pool:
                ans.append([keystroke])
            return ans

        token_pool = set(
            [
                token
                for token in self.token_pool
                if self.get_token_distance(token) != float("inf")
            ]
        )
        result = dp_search(keystroke, token_pool)
        if not result:
            token_pool = set([token for token in self.token_pool])
            result = dp_search(keystroke, token_pool)

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
        for word in self.total_composition_words:
            if len(word) == 1 and is_chinese_character(word):
                if not self._user_frequency_db.word_exists(word):
                    self._user_frequency_db.insert(None, word, 1)
                else:
                    self._user_frequency_db.increment_word_frequency(word)

    def update_user_phrase_db(self, text: str) -> None:
        raise NotImplementedError("update_user_phrase_db is not implemented yet")

    def new_reconstruct(self, keystroke: str, top_n: int = 5) -> list[list[str]]:
        if not keystroke:
            return []

        # Get all possible seps
        possible_seps = []
        for ime_type in self.activated_imes:
            token_ways = self.ime_handlers[ime_type].tokenize(keystroke)
            possible_seps.extend(token_ways)

        # Filter out empty sep
        possible_seps = [sep for sep in possible_seps if sep]
        # Filter out same sep
        possible_seps = [list(t) for t in set(tuple(token) for token in possible_seps)]

        token_pool = set([token for sep in possible_seps for token in sep])
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

        self.logger.info("Creating Graph with %d possible seps", len(new_possible_seps))

        sentence_graph = SentenceGraph()
        for sep_tokens in new_possible_seps:
            sep_tokens = [
                (token, self.get_token_distance(token)) for token in sep_tokens
            ]
            sentence_graph.add_token_path(sep_tokens)

        possible_paths = sentence_graph.get_sentence()
        return possible_paths[:top_n]

    def old_phase1(self, keystroke: str) -> list[str]:
        self.unfreeze_keystrokes = keystroke
        self._update_token_pool()
        possible_sentences = self._reconstruct_sentence_from_pre_possible_sentences(
            self.unfreeze_keystrokes
        )
        possible_sentences = self._sort_possible_sentences(possible_sentences)
        return possible_sentences

    def end_to_end(self, keystroke: str) -> list[str]:
        token_sentences = self.new_reconstruct(keystroke)
        if not token_sentences:
            return []
        return self._token_sentence_to_word_sentence(token_sentences[0])


if __name__ == "__main__":
    handler = KeyEventHandler()
    test_case = "soshite baai hello world"
    phase1_result = handler.old_phase1(test_case)
    new_result = handler.new_reconstruct(test_case)
    print("---------------------")
    print("PHASE1", phase1_result)
    print("NEW", new_result)
    print("---------------------")
    print("PHASE1", handler._calculate_sentence_distance(phase1_result[0]))
    print("NEW", handler._calculate_sentence_distance(new_result[0]))
    print("---------------------")
    print("PHASE1", handler.end_to_end(test_case))
    print("NEW", handler.end_to_end(test_case))
    print("---------------------")
