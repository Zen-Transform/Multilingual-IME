def is_chinese_character(char):
    return any([
        '\u4e00'  <= char <= '\u9fff',  # CJK Unified Ideographs
        '\u3400'  <= char <= '\u4dbf',  # CJK Unified Ideographs Extension A
        '\u20000' <= char <= '\u2a6df', # CJK Unified Ideographs Extension B
        '\u2a700' <= char <= '\u2b73f', # CJK Unified Ideographs Extension C
        '\u2b740' <= char <= '\u2b81f', # CJK Unified Ideographs Extension D
        '\uf900'  <= char <= '\ufaff',  # CJK Compatibility Ideographs
    ])

def is_all_chinese_char(text):
    return all([is_chinese_character(char) for char in text])

def is_english_char(char):
    return any([
        '\u0041' <= char <= '\u005a',  # Uppercase
        '\u0061' <= char <= '\u007a',  # Lowercase
    ])

def is_number(char):
    return any([
        '\u0030' <= char <= '\u0039',  # 0-9
    ])

def is_punctuation(char):
    return any([
        '\u0021' <= char <= '\u002f',  # !"#$%&'()*+,-./
        '\u003a' <= char <= '\u0040',  # :;<=>?@
        '\u005b' <= char <= '\u0060',  # [\]^_`
        '\u007b' <= char <= '\u007e',  # {|}~
    ])

def is_chinese_punctuation(char):
    return any([
        '\u3000' <= char <= '\u303f',  # CJK Symbols and Punctuation
    ])

