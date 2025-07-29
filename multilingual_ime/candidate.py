from colorama import Fore, Style


class Candidate:
    """
    Represents a candidate word in an input method editor (IME) system.
    Attributes:
        word (str): The candidate word.
        keystrokes (str): The keystrokes associated with the candidate word.
        word_frequency (int | str): The frequency of the word, can be an integer or a string.
        user_key (str): The user key associated with the candidate.
        distance (int | str): The distance metric for the candidate, can be an integer or a string.
        ime_method (str): The input method used for this candidate.
    """

    def __init__(
        self,
        word: str,
        keystrokes: str,
        word_frequency: int,
        user_key: str,
        distance: int,
        method: str,
    ):
        self.__word = word
        self.__keystrokes = keystrokes
        self.__word_frequency = word_frequency
        self.__user_key = user_key
        self.__distance = distance
        self.__ime_method = method

    @property
    def word(self):
        """
        Returns the candidate word.
        """
        return self.__word

    @property
    def keystrokes(self):
        """
        Returns the keystrokes associated with the candidate word.
        """
        return self.__keystrokes

    @property
    def word_frequency(self):
        """
        Returns the word frequency of the candidate word.
        """
        return self.__word_frequency

    @property
    def user_key(self):
        """
        Returns the original user key associated with the candidate word.
        """
        return self.__user_key

    @property
    def distance(self):
        """
        Returns the distance of the candidate word and the user key.
        This can be used to determine how closely the candidate matches the user's input.
        """
        return self.__distance

    @property
    def ime_method(self):
        """
        Returns the input method of the candidate word.
        This indicates which input method was used to generate the candidate.
        """
        return self.__ime_method

    def to_dict(self) -> dict:
        """
        Converts the candidate object to a dictionary representation.

        Returns:
            dict: A dictionary representation of the candidate object.
        """
        return {
            "word": self.word,
            "keystrokes": self.keystrokes,
            "word_frequency": self.word_frequency,
            "user_key": self.user_key,
            "distance": self.distance,
            "ime_method": self.ime_method,
        }

    def __repr__(self):
        result = ""
        if self.ime_method == "english":
            result = f"{Fore.GREEN}{self.word}{Style.RESET_ALL}"
        elif self.ime_method == "cangjie":
            result = f"{Fore.YELLOW}{self.word}{Style.RESET_ALL}"
        elif self.ime_method == "bopomofo":
            result = f"{Fore.CYAN}{self.word}{Style.RESET_ALL}"
        elif self.ime_method == "pinyin":
            result = f"{Fore.MAGENTA}{self.word}{Style.RESET_ALL}"
        else:
            result = f"{Fore.WHITE}{self.word}{Style.RESET_ALL}"

        if self.distance != 0:
            result = f"{Fore.RED}{self.word}{Style.RESET_ALL}"

        return result


if __name__ == "__main__":
    cand = Candidate("word", "keystrokes", 1, "user_key", 1, "method")
    cand1 = Candidate("word", "keystrokes", 1, "user_key", 1, "method")
    print(cand.to_dict())
    print(cand1.to_dict())
