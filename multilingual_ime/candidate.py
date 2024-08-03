class CandidateWord():
    def __init__(self, word, keystrokes, word_frequency):
        self.word = word
        self.keystrokes = keystrokes
        self.word_frequency = word_frequency
        self.user_key = None
        self.distance = None
    
    def to_dict(self):
        return {
            "word": self.word,
            "keystrokes": self.keystrokes,
            "word_frequency": self.word_frequency,
            "user_key": self.user_key,
            "distance": self.distance
        }
