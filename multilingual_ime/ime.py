import re
from pathlib import Path
from .core.custom_decorators import lru_cache_with_doc
from itertools import chain
from abc import ABC, abstractmethod

from .ime_detector import IMETokenDetectorDL
from .keystroke_map_db import KeystrokeMappingDB

# Define the IME names
BOPOMOFO_IME = "bopomofo"
CANGJIE_IME = "cangjie"
PINYIN_IME = "pinyin"
ENGLISH_IME = "english"


# Define IME DB paths
BOPOMOFO_IME_DB_PATH = Path(__file__).parent / "src" / "bopomofo_keystroke_map.db"
CANGJIE_IME_DB_PATH = Path(__file__).parent / "src" / "cangjie_keystroke_map.db"
PINYIN_IME_DB_PATH = Path(__file__).parent / "src" / "pinyin_keystroke_map.db"
ENGLISH_IME_DB_PATH = Path(__file__).parent / "src" / "english_keystroke_map.db"

# Define IME token detector model paths

BOPOMOFO_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent
    / "src"
    / "models"
    / "one_hot_dl_token_model_bopomofo_2024-10-27.pth"
)
CANGJIE_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent
    / "src"
    / "models"
    / "one_hot_dl_token_model_cangjie_2024-10-27.pth"
)
PINYIN_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent
    / "src"
    / "models"
    / "one_hot_dl_token_model_pinyin_2024-10-27.pth"
)
ENGLISH_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent
    / "src"
    / "models"
    / "one_hot_dl_token_model_english_2024-10-27.pth"
)


class IME(ABC):
    def __init__(self):
        self.token_detector: IMETokenDetectorDL = None
        self.keystroke_map_db: KeystrokeMappingDB = None

    @abstractmethod
    def tokenize(self, keystroke: str) -> list[list[str]]:
        pass

    def get_token_candidates(self, token: str) -> list[tuple[str, str, int]]:
        return self.keystroke_map_db.get_closest(token)

    def get_closest_word_distance(self, token: str) -> list[tuple[str, str, int]]:
        return self.keystroke_map_db.closest_word_distance(token)

    def is_valid_token(self, keystroke: str) -> bool:
        return self.token_detector.predict(keystroke)

    def closest_word_distance(self, keystroke: str) -> int:
        return self.keystroke_map_db.closest_word_distance(keystroke)
    

class BopomofoIME(IME):
    def __init__(self):
        super().__init__()
        self.token_detector = IMETokenDetectorDL(
            model_path=BOPOMOFO_IME_TOKEN_DETECTOR_MODEL_PATH,
            device="cpu",
            verbose_mode=False,
        )
        self.keystroke_map_db = KeystrokeMappingDB(db_path=BOPOMOFO_IME_DB_PATH)

    def tokenize(self, keystroke: str) -> list[list[str]]:
        def cut_bopomofo_with_regex(bopomofo_keystroke: str) -> list[str]:
            if not bopomofo_keystroke:
                return []
            tokens = re.split(r"(?<=3|4|6|7| )", bopomofo_keystroke)
            ans = [token for token in tokens if token]
            return ans

        if not keystroke:
            return []

        bopomofo_tokens = cut_bopomofo_with_regex(keystroke)
        assert (
            "".join(bopomofo_tokens) == keystroke
        ), f"Error: {__class__}.tokenize failed, keystroke'{keystroke}' mismatch with {bopomofo_tokens}"
        return [bopomofo_tokens]


class CangjieIME(IME):
    def __init__(self):
        super().__init__()
        self.token_detector = IMETokenDetectorDL(
            model_path=CANGJIE_IME_TOKEN_DETECTOR_MODEL_PATH,
            device="cpu",
            verbose_mode=False,
        )
        self.keystroke_map_db = KeystrokeMappingDB(db_path=CANGJIE_IME_DB_PATH)

    def tokenize(self, keystroke: str) -> list[list[str]]:
        # TODO: Implement cangjie tokenizer with DP

        def cut_cangjie_with_regex(cangjie_keystroke: str) -> list[str]:
            if not cangjie_keystroke:
                return []
            tokens = re.split(r"(?<=[ ])", cangjie_keystroke)
            ans = [token for token in tokens if token]
            return ans

        if not keystroke:
            return []

        cangjie_tokens = cut_cangjie_with_regex(keystroke)
        assert "".join(cangjie_tokens) == keystroke
        return [cangjie_tokens]


with open(
    Path(__file__).parent / "src" / "intact_pinyin.txt", "r", encoding="utf-8"
) as f:
    intact_pinyin_set = set(s for s in f.read().split("\n"))

special_characters = " !@#$%^&*()-_=+[]{}|;:'\",.<>?/`~"
sepcial_char_set = [c for c in special_characters]
intact_pinyin_set = intact_pinyin_set.union(sepcial_char_set)

# Add special characters, since they will be separated individually

all_pinyin_set = set(s[:i] for s in intact_pinyin_set for i in range(1, len(s) + 1))

intact_cut_pinyin_ans = {}
all_cut_pinyin_ans = {}


class PinyinIME(IME):
    def __init__(self):
        super().__init__()
        self.token_detector = IMETokenDetectorDL(
            model_path=PINYIN_IME_TOKEN_DETECTOR_MODEL_PATH,
            device="cpu",
            verbose_mode=False,
        )
        self.keystroke_map_db = KeystrokeMappingDB(db_path=PINYIN_IME_DB_PATH)

    def tokenize(self, keystroke: str) -> list[list[str]]:
        # Modified from https://github.com/OrangeX4/simple-pinyin.git

        @lru_cache_with_doc(maxsize=128, typed=False)
        def cut_pinyin(pinyin: str, is_intact: bool = False) -> list[list[str]]:
            if is_intact:
                pinyin_set = intact_pinyin_set
            else:
                pinyin_set = all_pinyin_set

            if pinyin in pinyin_set:
                return [[pinyin]]

            # If result is not in the word set, DP by recursion
            ans = []
            for i in range(1, len(pinyin)):
                # If pinyin[:i], is a right word, continue DP
                if pinyin[:i] in pinyin_set:
                    former = [pinyin[:i]]
                    appendices_solutions = cut_pinyin(pinyin[i:], is_intact)
                    for appendixes in appendices_solutions:
                        ans.append(former + appendixes)
            if ans == []:
                return [[pinyin]]
            return ans

        def cut_pinyin_with_error_correction(pinyin: str) -> list[str]:
            ans = {}
            for i in range(1, len(pinyin) - 1):
                key = pinyin[:i] + pinyin[i + 1] + pinyin[i] + pinyin[i + 2 :]
                key_ans = cut_pinyin(key, is_intact=True)
                if key_ans:
                    ans[key] = key_ans
            return list(chain.from_iterable(ans.values()))

        if not keystroke:
            return []

        total_ans = []
        total_ans.extend(cut_pinyin(keystroke, is_intact=True))
        # total_ans.extend(cut_pinyin(keystroke, is_intact=False))
        for ans in total_ans:
            assert "".join(ans) == keystroke
        # total_ans.extend(cut_pinyin_with_error_correction(keystroke))

        return total_ans


class EnglishIME(IME):
    def __init__(self):
        super().__init__()
        self.token_detector = IMETokenDetectorDL(
            model_path=ENGLISH_IME_TOKEN_DETECTOR_MODEL_PATH,
            device="cpu",
            verbose_mode=False,
        )
        self.keystroke_map_db = KeystrokeMappingDB(db_path=ENGLISH_IME_DB_PATH)

    def tokenize(self, keystroke: str) -> list[list[str]]:
        def cut_english(english_keystroke: str) -> list[str]:
            if not english_keystroke:
                return []
            tokens = re.split(r"(\s|[^\w])", english_keystroke)
            ans = [token for token in tokens if token]
            return ans

        if not keystroke:
            return []

        english_tokens = cut_english(keystroke)
        assert "".join(english_tokens) == keystroke
        return [english_tokens]


class IMEFactory:
    @staticmethod
    def create_ime(ime_type: str) -> IME:
        if ime_type == BOPOMOFO_IME:
            return BopomofoIME()
        if ime_type == CANGJIE_IME:
            return CangjieIME()
        if ime_type == PINYIN_IME:
            return PinyinIME()
        if ime_type == ENGLISH_IME:
            return EnglishIME()
        else:
            raise ValueError(f"IME type {ime_type} not supported")
