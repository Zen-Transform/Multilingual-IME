import joblib
from abc import ABC, abstractmethod

from colorama import Fore, Style


# from SVMClassification import custom_tokenizer_bopomofo, custom_tokenizer_cangjie, custom_tokenizer_pinyin # fix this depends on SVMClassification.py


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
    my_bopomofo_detector = IMEDetectorSVM('..\\Model_dump\\bopomofo_8adj.pkl', '..\\Model_dump\\vectorizer_bopomofo_8adj.pkl')
    my_eng_detector = IMEDetectorSVM('..\\Model_dump\\english_8adj.pkl', '..\\Model_dump\\vectorizer_english_8adj.pkl')
    my_cangjie_detector = IMEDetectorSVM('..\\Model_dump\\cangjie_8adj.pkl', '..\\Model_dump\\vectorizer_cangjie_8adj.pkl')
    my_pinyin_detector = IMEDetectorSVM('..\\Model_dump\\pinyin_8adj.pkl', '..\\Model_dump\\vectorizer_pinyin_8adj.pkl')
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
