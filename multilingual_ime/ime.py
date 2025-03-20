import re
from pathlib import Path
from itertools import chain
from abc import ABC, abstractmethod

from .ime_detector import IMETokenDetectorDL
from .keystroke_map_db import KeystrokeMappingDB
from .core.custom_decorators import lru_cache_with_doc

# Define the IME names
BOPOMOFO_IME = "bopomofo"
CANGJIE_IME = "cangjie"
PINYIN_IME = "pinyin"
ENGLISH_IME = "english"
SPECIAL_IME = "special"
JAPANESE_IME = "japanese"


# Define IME DB paths
BOPOMOFO_IME_DB_PATH = Path(__file__).parent / "src" / "bopomofo_keystroke_map.db"
CANGJIE_IME_DB_PATH = Path(__file__).parent / "src" / "cangjie_keystroke_map.db"
PINYIN_IME_DB_PATH = Path(__file__).parent / "src" / "pinyin_keystroke_map.db"
ENGLISH_IME_DB_PATH = Path(__file__).parent / "src" / "english_keystroke_map.db"
SPECIAL_IME_DB_PATH = (
    Path(__file__).parent / "src" / "special_character_keystroke_map.db"
)
JAPANESE_IME_DB_PATH = Path(__file__).parent / "src" / "japanese_keystroke_map.db"

# Define IME valid keystroke set
BOPOMOFO_VALID_KEYSTROKE_SET = set("1qaz2wsx3edc4rfv5tgb6yhn7ujm8ik,9ol.0p;/- ")
CANGJIE_VALID_KEYSTROKE_SET = set(" qwertyuiopasdfghjklzxcvbnm")
PINYIN_VALID_KEYSTROKE_SET = set(" abcdefghijklmnopqrstuvwxyz")
ENGLISH_VALID_KEYSTROKE_SET = set(
    " abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
JAPANESE_VALID_KEYSTROKE_SET = set(" abcdefghijklmnopqrstuvwxyz")

# Define IME token length
BOPOMOFO_IME_MIN_TOKEN_LENGTH = 2
BOPOMOFO_IME_MAX_TOKEN_LENGTH = 4
CANGJIE_IME_MIN_TOKEN_LENGTH = 2
CANGJIE_IME_MAX_TOKEN_LENGTH = 5
PINYIN_IME_MIN_TOKEN_LENGTH = 1
PINYIN_IME_MAX_TOKEN_LENGTH = 6
ENGLISH_IME_MIN_TOKEN_LENGTH = 1
ENGLISH_IME_MAX_TOKEN_LENGTH = 30  # FIXME: Need to be confirmed (what is the max length of 80% frequency used english word), 30 is a random number
JAPANESE_IME_MIN_TOKEN_LENGTH = 1
JAPANESE_IME_MAX_TOKEN_LENGTH = 30  # FIXME: Need to be confirmed, kanji can be long


# Define IME token length variance (for case of user input additional keystroke)
IME_TOKEN_LENGTH_VARIANCE = 1

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
JAPANESE_IME_TOKEN_DETECTOR_MODEL_PATH = (
    Path(__file__).parent
    / "src"
    / "models"
    / "one_hot_dl_token_model_japanese_2025-03-20.pth"
)


class IME(ABC):
    def __init__(self):
        self.token_detector: IMETokenDetectorDL = None
        self.keystroke_map_db: KeystrokeMappingDB = None

    @abstractmethod
    def tokenize(self, keystroke: str) -> list[list[str]]:
        pass

    @abstractmethod
    def string_to_keystroke(self, string: str) -> str:
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
        assert "".join(bopomofo_tokens) == keystroke, (
            "Error: {__class__}.tokenize failed, "
            f"keystroke'{keystroke}' mismatch with {bopomofo_tokens}"
        )
        return [bopomofo_tokens]

    def is_valid_token(self, keystroke):
        if (
            len(keystroke) < BOPOMOFO_IME_MIN_TOKEN_LENGTH - IME_TOKEN_LENGTH_VARIANCE
            or len(keystroke)
            > BOPOMOFO_IME_MAX_TOKEN_LENGTH + IME_TOKEN_LENGTH_VARIANCE
        ):
            return False
        if any(c not in BOPOMOFO_VALID_KEYSTROKE_SET for c in keystroke):
            return False
        return super().is_valid_token(keystroke)

    def string_to_keystroke(self, string: str) -> str:
        raise NotImplementedError("BopomofoIME does not support string_to_keystroke")


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

    def is_valid_token(self, keystroke):
        if (
            len(keystroke) < CANGJIE_IME_MIN_TOKEN_LENGTH - IME_TOKEN_LENGTH_VARIANCE
            or len(keystroke) > CANGJIE_IME_MAX_TOKEN_LENGTH + IME_TOKEN_LENGTH_VARIANCE
        ):
            return False
        if any(c not in CANGJIE_VALID_KEYSTROKE_SET for c in keystroke):
            return False
        return super().is_valid_token(keystroke)

    def string_to_keystroke(self, string: str) -> str:
        raise NotImplementedError("CangjieIME does not support string_to_keystroke")


with open(
    Path(__file__).parent / "src" / "intact_pinyin.txt", "r", encoding="utf-8"
) as f:
    intact_pinyin_set = set(s for s in f.read().split("\n"))

special_characters = " !@#$%^&*()-_=+[]{}|;:'\",.<>?/`~"
special_char_set = [c for c in special_characters]
intact_pinyin_set = intact_pinyin_set.union(special_char_set)

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

    def is_valid_token(self, keystroke):
        if (
            len(keystroke) < PINYIN_IME_MIN_TOKEN_LENGTH - IME_TOKEN_LENGTH_VARIANCE
            or len(keystroke) > PINYIN_IME_MAX_TOKEN_LENGTH + IME_TOKEN_LENGTH_VARIANCE
        ):
            return False
        if any(c not in PINYIN_VALID_KEYSTROKE_SET for c in keystroke):
            return False
        return super().is_valid_token(keystroke)

    def string_to_keystroke(self, string: str) -> str:
        raise NotImplementedError("PinyinIME does not support string_to_keystroke")


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

    def is_valid_token(self, keystroke):
        if (
            len(keystroke) < ENGLISH_IME_MIN_TOKEN_LENGTH - IME_TOKEN_LENGTH_VARIANCE
            or len(keystroke) > ENGLISH_IME_MAX_TOKEN_LENGTH + IME_TOKEN_LENGTH_VARIANCE
        ):
            return False
        if keystroke == " ":
            return True
        if len(keystroke) > 2 and " " in keystroke:
            return False
        if any(c not in ENGLISH_VALID_KEYSTROKE_SET for c in keystroke):
            return False
        return super().is_valid_token(keystroke)

    def string_to_keystroke(self, string: str) -> str:
        raise NotImplementedError("EnglishIME does not support string_to_keystroke")


class SpecialCharacterIME(IME):
    def __init__(self):
        super().__init__()
        self.keystroke_map_db = KeystrokeMappingDB(db_path=SPECIAL_IME_DB_PATH)

    def tokenize(self, keystroke: str) -> list[list[str]]:
        result = []
        i = 0
        while i < len(keystroke):
            if keystroke[i] == "©":
                result.append("©" + keystroke[i + 1])
                i += 2
            else:
                i += 1
        return [result]

    def is_valid_token(self, keystroke):
        if keystroke.startswith("©") and len(keystroke) == 2:
            return True
        return False

    def string_to_keystroke(self, string: str) -> str:
        raise NotImplementedError(
            "SpecialCharacterIME does not support string_to_keystroke"
        )


with open(
    Path(__file__).parent / "src" / "intact_japanese.txt", "r", encoding="utf-8"
) as f:
    intact_japanese_set = set(s for s in f.read().split("\n"))


class JapaneseIME(IME):
    def __init__(self):
        super().__init__()
        self.token_detector = IMETokenDetectorDL(
            model_path=JAPANESE_IME_TOKEN_DETECTOR_MODEL_PATH,
            device="cpu",
            verbose_mode=False,
        )
        self.keystroke_map_db = KeystrokeMappingDB(db_path=JAPANESE_IME_DB_PATH)

    def tokenize(self, keystroke: str) -> list[list[str]]:
        def regex_split(input_str, delimiters):
            pattern = "|".join(
                sorted(map(re.escape, delimiters), key=len, reverse=True)
            )
            ans = [token for token in re.split(f"({pattern})", input_str) if token]
            return ans

        def group_japanese_tokens(japanese_tokens):
            new_japanese_tokens = []
            pre_token = ""
            for token in japanese_tokens:
                if token in intact_japanese_set:
                    if pre_token:
                        new_japanese_tokens.append(pre_token)
                    new_japanese_tokens.append(token)
                    pre_token = ""
                else:
                    pre_token += token
            if pre_token:
                new_japanese_tokens.append(pre_token)
            return new_japanese_tokens

        if not keystroke:
            return []
        japanese_tokens = regex_split(keystroke, list(intact_japanese_set))
        japanese_tokens = group_japanese_tokens(japanese_tokens)
        assert "".join(japanese_tokens) == keystroke
        return [japanese_tokens]

    def is_valid_token(self, keystroke):
        if any(c not in ENGLISH_VALID_KEYSTROKE_SET for c in keystroke):
            return False
        if keystroke in intact_japanese_set:
            return True
        return super().is_valid_token(keystroke)

    def string_to_keystroke(self, string: str) -> str:
        raise NotImplementedError("JapaneseIME does not support string_to_keystroke")
        #  TODO: Implement string_to_keystroke by


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
        if ime_type == SPECIAL_IME:
            return SpecialCharacterIME()
        if ime_type == JAPANESE_IME:
            return JapaneseIME()
        raise ValueError(f"IME type {ime_type} not supported")
