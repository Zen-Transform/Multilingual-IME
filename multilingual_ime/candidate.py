class CandidateWord:
    def __init__(self, word, keystrokes, word_frequency):
        self.word = word
        self.keystrokes = keystrokes
        self.word_frequency = word_frequency
        self.user_key = None
        self.distance = None
        self.method = None

    def to_dict(self):
        return {
            "word": self.word,
            "keystrokes": self.keystrokes,
            "word_frequency": self.word_frequency,
            "user_key": self.user_key,
            "distance": self.distance,
        }

    def set_method(self, method: str):
        self.method = method
        return self


class Candidate:
    def __init__(self, word, keystrokes, word_frequency, user_key, distance, method):
        self.__word = word
        self.__keystrokes = keystrokes
        self.__word_frequency = word_frequency
        self.__user_key = user_key
        self.__distance = distance
        self.__ime_method = method

    @property
    def word(self):
        return self.__word

    @property
    def keystrokes(self):
        return self.__keystrokes

    @property
    def word_frequency(self):
        return self.__word_frequency

    @property
    def user_key(self):
        return self.__user_key

    @property
    def distance(self):
        return self.__distance

    @property
    def ime_method(self):
        return self.__ime_method

    def to_dict(self):
        return {
            "word": self.word,
            "keystrokes": self.keystrokes,
            "word_frequency": self.word_frequency,
            "user_key": self.user_key,
            "distance": self.distance,
            "ime_method": self.ime_method,
        }

    # def __setattr__(self, name, value):
    #     if hasattr(self, name):
    #         raise AttributeError(f"Cannot set attribute {name} to {value}, {__class__.__name__} is an immutable object")
    
if __name__ == "__main__":
    cand = Candidate("word", "keystrokes", 1, "user_key", 1, "method")
    cand1 = Candidate("word", "keystrokes", 1, "user_key", 1, "method")
    print(cand.to_dict())
    print(cand1.to_dict())
    cand1.word = "new_word1"
    cand.word = "new_word"

