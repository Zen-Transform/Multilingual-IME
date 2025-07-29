"""
This module provides utility functions for string manipulation,
including calculating distances between strings, checking character types,
and identifying Chinese and English characters and punctuation.
"""

def modified_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the modified Levenshtein distance between two strings.
    The modified Levenshtein distance is the minimum number of single-character \
    edits (insertions, deletions, substitutions, or swaps) required to change \
    one word into the other.
    The difference between the modified Levenshtein distance and the original \
    Levenshtein distance is that the modified Levenshtein distance allows for \
    swapping two adjacent characters.
    
    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The modified Levenshtein distance between the two strings.

    Examples:
        >>> modified_levenshtein_distance("abc", "abcc")  # 1
        >>> modified_levenshtein_distance("abc", "acb")  # 1
        >>> modified_levenshtein_distance("flaw", "lawn")  # 2
        >>> modified_levenshtein_distance("kitten", "sitting")  # 3
    """

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, char1 in enumerate(s1):
        current_row = [i + 1]

        for j, char2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            # substitutions = previous_row[j] + (char1 != char2)
            if char1 != char2:
                if i > 0 and j > 0 and s1[i - 1] == char2 and s1[i] == char1:
                    substitutions = previous_row[j - 1]
                else:
                    substitutions = previous_row[j] + 1
            else:
                substitutions = previous_row[j]

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    The Levenshtein distance is the minimum number of \
    single-character edits (insertions, deletions, or substitutions) \
    required to change one word into the other.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings.

    Examples:
        >>> levenshtein_distance("abc", "abcc")  # 1
        >>> levenshtein_distance("abc", "acb")  # 2
        >>> levenshtein_distance("flaw", "lawn")  # 2
        >>> levenshtein_distance("kitten", "sitting")  # 3
    """

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, char1 in enumerate(s1):
        current_row = [i + 1]

        for j, char2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char1 != char2)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def is_chinese_character(char: str) -> bool:
    """
    Check if a character is a Chinese character.
    Args:
        char (str): The character to check.
    Returns:
        bool: True if the character is a Chinese character, False otherwise.

    """
    return any(
        [
            "\u4e00" <= char <= "\u9fff",  # CJK Unified Ideographs
            "\u3400" <= char <= "\u4dbf",  # CJK Unified Ideographs Extension A
            "\u20000" <= char <= "\u2a6df",  # CJK Unified Ideographs Extension B
            "\u2a700" <= char <= "\u2b73f",  # CJK Unified Ideographs Extension C
            "\u2b740" <= char <= "\u2b81f",  # CJK Unified Ideographs Extension D
            "\uf900" <= char <= "\ufaff",  # CJK Compatibility Ideographs
        ]
    )


def is_all_chinese_char(text: str) -> bool:
    """
    Check if all characters in a string are Chinese characters.
    Args:
        text (str): The string to check.
    Returns:
        bool: True if all characters are Chinese characters, False otherwise.
    """
    return all(is_chinese_character(char) for char in text)


def is_english_char(char: str) -> bool:
    """
    Check if a character is an English letter.
    Args:
        char (str): The character to check.
    Returns:
        bool: True if the character is an English letter, False otherwise.
    """
    return any(
        [
            "\u0041" <= char <= "\u005a",  # Uppercase
            "\u0061" <= char <= "\u007a",  # Lowercase
        ]
    )


def is_number(char: str) -> bool:
    """
    Check if a character is a number.
    Args:
        char (str): The character to check.
    Returns:
        bool: True if the character is a number, False otherwise.
    """
    return any(
        [
            "\u0030" <= char <= "\u0039",  # 0-9
        ]
    )


def is_english_punctuation(char: str) -> bool:
    """
    Check if a character is a English punctuation mark.
    Args:
        char (str): The character to check.
    Returns:
        bool: True if the character is a punctuation mark, False otherwise.
    """
    return any(
        [
            "\u0021" <= char <= "\u002f",  # !"#$%&'()*+,-./
            "\u003a" <= char <= "\u0040",  # :;<=>?@
            "\u005b" <= char <= "\u0060",  # [\]^_`
            "\u007b" <= char <= "\u007e",  # {|}~
        ]
    )


def is_chinese_punctuation(char: str) -> bool:
    """
    Check if a character is a Chinese punctuation mark.
    Args:
        char (str): The character to check.
    Returns:
        bool: True if the character is a Chinese punctuation mark, False otherwise.
    """
    return any(
        [
            "\u3000" <= char <= "\u303f",  # CJK Symbols and Punctuation
        ]
    )
