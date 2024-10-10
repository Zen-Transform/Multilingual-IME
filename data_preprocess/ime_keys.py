
class IMEKeys():
    bopomofo_map = {
                "ㄅ": "1", "ㄆ": "q", "ㄇ": "a",
                "ㄈ": "z", "ㄉ": "2", "ㄊ": "w",
                "ㄋ": "s", "ㄌ": "x", "ㄍ": "e",
                "ㄎ": "d", "ㄏ": "c", "ㄐ": "r",
                "ㄑ": "f", "ㄒ": "v", "ㄓ": "5",
                "ㄔ": "t", "ㄕ": "g", "ㄖ": "b",
                "ㄗ": "y", "ㄘ": "h", "ㄙ": "n",
                "ㄚ": "8", "ㄛ": "i", "ㄜ": "k",
                "ㄝ": ",", "ㄞ": "9", "ㄟ": "o",
                "ㄠ": "l", "ㄡ": ".", "ㄢ": "0",
                "ㄣ": "p", "ㄤ": ";", "ㄥ": "/",
                "ㄦ": "-", "ㄧ": "u", "ㄨ": "j",
                "ㄩ": "m", "ˊ": "6", "ˇ": "3",
                "ˋ": "4",  "˙": "7", " ": " "
            }
    cangjie_map = {"手": "q", "田": "w", "水": "e",
               "口": "r", "廿": "t", "卜": "y",
               "山": "u", "戈": "i", "人": "o",
               "心": "p", "日": "a", "尸": "s",
               "木": "d", "火": "f", "土": "g",
               "竹": "h", "十": "j", "大": "k",
               "中": "l", "重": "z", "難": "x",
               "金": "c", "女": "v", "月": "b",
               "弓": "n", "一": "m", " ": " "
            }
    pinyin_map = {  # fix: to be confirmed
        # define initial 
        "b": "b", "p": "p", "m": "m",  
        "f": "f", "d": "d", "t": "t",
        "n": "n", "l": "l", "g": "g",
        "k": "k", "h": "h", "j": "j",
        "q": "q", "x": "x", "zh": "zh",
        "ch": "ch", "sh": "sh", "r": "r",
        "z": "z", "c": "c", "s": "s",
        "y": "y", "w": "w",
        # define vowels
        "a": "a", "o": "o", "e": "e",
        "i": "i", "u": "u", "ai": "ai",
        "ei": "ei", "ui": "ui", "ao": "ao",
        "ou": "ou", "ie": "ie", "ue": "ue",
        "er": "er", "an": "an", "en": "en",
        "in": "in", "un": "un", "ang": "ang",
        "eng": "eng", "ing": "ing", "ong": "ong",
        # 認讀音節
        "zhi": "zhi", "chi": "chi", "shi": "shi",
        "ri": "ri", "zi": "zi", "ci": "ci",
        "si": "si", "yi": "yi", "wu": "wu",
        "yu": "yu", "ye": "ye", "yue": "yue",
        "yin": "yin", "yun": "yun", "yuan": "yuan",
        "ying": "ying"
    }
    bopomofo_keys = set([v for k, v in bopomofo_map.items()])
    cangjie_keys = set([v for k, v in cangjie_map.items()])
    pinyin_keys = set("abcdefghijklmnopqrstuvwxyz")  # fix: to be confirmed

    english_keys = set("1234567890-=" + \
                   "qwertyuiop[]" + \
                   "asdfghjkl;'" + \
                   "zxcvbnm,./ " + \
                   "!@#$%^&*()_+" + \
                   "QWERTYUIOP{}" + \
                   "ASDFGHJKL:\"" + \
                   "ZXCVBNM<>? ")
    
    universal_keys = set("`1234567890-=" + \
                     "qwertyuiop[]\\" + \
                     "asdfghjkl;'" + \
                     "zxcvbnm,./" + \
                     "~!@#$%^&*()_+" + \
                     "QWERTYUIOP{}|" + \
                     "ASDFGHJKL:\"" + \
                     "ZXCVBNM<>?" + \
                     " ")
    contrl_keys = "®"

    @classmethod
    def is_bopomofo(cls, keystroke: str) -> bool:
        for key in keystroke:
            if key not in cls.bopomofo_keys:
                return False
        return True
    
    @classmethod
    def is_cangjie(cls, keystroke: str) -> bool:
        for key in keystroke:
            if key not in cls.cangjie_keys:
                return False
        return True
    
    @classmethod
    def is_pinyin(cls, keystroke: str) -> bool:
        for key in keystroke:
            if key not in cls.pinyin_keys:
                return False
        return True
    
    @classmethod
    def is_english(cls, keystroke: str) -> bool:
        for key in keystroke:
            if key not in cls.english_keys:
                return False
        return True
    
    @classmethod
    def is_ime_keystroke(cls, keystroke: str, ime_type: str) -> bool:
        if ime_type == "bopomofo":
            return cls.is_bopomofo(keystroke)
        elif ime_type == "cangjie":
            return cls.is_cangjie(keystroke)
        elif ime_type == "pinyin":
            return cls.is_pinyin(keystroke)
        elif ime_type == "english":
            return cls.is_english(keystroke)
        else:
            raise ValueError("Invalid ime_type: " + ime_type)
    