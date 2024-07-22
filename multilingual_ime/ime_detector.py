from abc import ABC, abstractmethod

import joblib
import torch
from colorama import Fore, Style

from .data_preprocess.keystroke_tokenizer import KeystrokeTokenizer

class IMEDetector(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def predict(self, input: str) -> str:
        pass

MAX_TOKEN_SIZE = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")

class IMEDetectorOneHot(IMEDetector):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._classifier = None
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        try:
            self._classifier = joblib.load(model_path)
            print(f'Model loaded from {model_path}')
            print(self._classifier)
        except Exception as e:
            print(f'Error loading model {model_path}')
            print(e)

    def _one_hot_encode(self, input_keystroke: str) -> torch.Tensor:
        token_ids = KeystrokeTokenizer.token_to_ids(KeystrokeTokenizer.tokenize(input_keystroke))
        token_ids = token_ids[:MAX_TOKEN_SIZE]  # truncate to MAX_TOKEN_SIZE 
        token_ids += [0] * (MAX_TOKEN_SIZE - len(token_ids))  # padding


        one_hot_keystrokes = torch.zeros(MAX_TOKEN_SIZE, KeystrokeTokenizer.key_labels_length()) \
                           + torch.eye(KeystrokeTokenizer.key_labels_length())[token_ids]
        one_hot_keystrokes = one_hot_keystrokes.view(-1)  # flatten
        return one_hot_keystrokes

    def predict(self, input_keystroke: str) -> bool:
        embedded_input = self._one_hot_encode(input_keystroke)
        embedded_input = embedded_input.to(DEVICE)
        self._classifier = self._classifier.to(DEVICE)

        with torch.no_grad():
            prediction = self._classifier(embedded_input)
            prediction = torch.argmax(prediction).item()
        return prediction
    

class IMEDetectorSVM(IMEDetector):
    def __init__(self, svm_model_path:str, tfidf_vectorizer_path:str) -> None:
        super().__init__()
        self.classifiers = None
        self.vectorizer = None
        self.load_model(svm_model_path, tfidf_vectorizer_path)


    def load_model(self, svm_model_path: str, tfidf_vectorizer_path:str) -> None:
        try:
            self.classifiers = joblib.load(svm_model_path)
            print(f'Model loaded from {svm_model_path}')
            self.vectorizer = joblib.load(tfidf_vectorizer_path)
            print(f'Vectorizer loaded from {tfidf_vectorizer_path}')

        except Exception as e:
            print(f'Error loading model and vectorizer.')
            print(e)


    def predict(self, input: str, positive_bound: float = 1, neg_bound: float = -0.5) -> bool:
        text_features = self.vectorizer.transform([input])
        predictions = {}
        for label, classifier in self.classifiers.items():
            prediction = classifier.decision_function(text_features)[0]
            predictions[label] = prediction

        if predictions["1"] > positive_bound or (neg_bound < predictions["1"] < 0):
            return True
        else:
            return False
    
    def predict_eng(self, input: str, positive_bound: float = 0.8, neg_bound: float = -0.7) -> bool:
        text_features = self.vectorizer.transform([input])
        predictions = {}
        for label, classifier in self.classifiers.items():
            prediction = classifier.decision_function(text_features)[0]
            predictions[label] = prediction

        if predictions["1"] > positive_bound or (neg_bound < predictions["1"] < 0):
            return True
        else:
            return False
        
    def predict_positive(self, input:str) -> float:
        text_features = self.vectorizer.transform([input])
        predictions = {}
        for label, classifier in self.classifiers.items():
            prediction = classifier.decision_function(text_features)[0]
            predictions[label] = prediction

        return predictions["1"]


if __name__ == "__main__":
    my_bopomofo_detector = IMEDetectorOneHot('multilingual_ime\\src\\model_dump\\one_hot_dl_model_bopomofo_2024-07-14.pkl')
    my_eng_detector = IMEDetectorOneHot('multilingual_ime\\src\\model_dump\\one_hot_dl_model_bopomofo_2024-07-14.pkl')
    my_cangjie_detector = IMEDetectorOneHot('multilingual_ime\\src\\model_dump\\one_hot_dl_model_bopomofo_2024-07-14.pkl')
    my_pinyin_detector = IMEDetectorOneHot('multilingual_ime\\src\\model_dump\\one_hot_dl_model_bopomofo_2024-07-14.pkl')
    input_text = "su3cl3"
    while True:
        input_text = input('Enter text: ')
        is_bopomofo = my_bopomofo_detector.predict(input_text)
        is_cangjie = my_cangjie_detector.predict(input_text)
        is_english = my_eng_detector.predict(input_text)
        is_pinyin = my_pinyin_detector.predict(input_text)

        print(Fore.GREEN + 'bopomofo'  if is_bopomofo else Fore.RED + 'bopomofo', end=' ')
        print(Fore.GREEN + 'cangjie' if is_cangjie else Fore.RED + 'cangjie', end=' ')
        print(Fore.GREEN + 'english' if is_english else Fore.RED + 'english', end=' ')
        print(Fore.GREEN + 'pinyin' if is_pinyin else Fore.RED + 'pinyin', end=' ')
        print(Style.RESET_ALL)
        print()
