import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import torch
from colorama import Fore, Style

from .keystroke_tokenizer import KeystrokeTokenizer
from .core.custom_decorators import deprecated
from .ime_phase1_train import LanguageDetectorModel

MAX_TOKEN_SIZE = 30


class IMEDetector(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def predict(self, input: str) -> str:
        pass


class IMEDetectorOneHot(IMEDetector):
    def __init__(
        self, model_path: str, device: str = "cuda", verbose_mode: bool = False
    ) -> None:
        super().__init__()
        self.logger.setLevel(logging.INFO if verbose_mode else logging.WARNING)
        self._classifier = None
        self._DEVICE = device

        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("cuda is not available, using cpu instead")
            device = "cpu"
        if not model_path.endswith(".pth"):
            self.logger.error("Invalid model path. Model must be a .pth file.")
            return

        self.load_model(model_path)
        self.logger.info(
            f"Detector created using the {self._DEVICE} device." if verbose_mode else ""
        )

    def load_model(self, model_path: str) -> None:
        try:
            self._classifier = LanguageDetectorModel(
                input_shape=MAX_TOKEN_SIZE * KeystrokeTokenizer.key_labels_length(),
                num_classes=2,
            )
            self._classifier.load_state_dict(
                torch.load(model_path, map_location=self._DEVICE, weights_only=True)
            )
            self.logger.info(f"Model loaded from {model_path}")
            self.logger.info(self._classifier)
        except Exception as e:
            self.logger.error(f"Error loading model {model_path}")
            self.logger.error(e)

    def _one_hot_encode(self, input_keystroke: str) -> torch.Tensor:
        token_ids = KeystrokeTokenizer.token_to_ids(
            KeystrokeTokenizer.tokenize(input_keystroke)
        )
        token_ids = token_ids[:MAX_TOKEN_SIZE]  # truncate to MAX_TOKEN_SIZE
        token_ids += [0] * (MAX_TOKEN_SIZE - len(token_ids))  # padding

        one_hot_keystrokes = (
            torch.zeros(MAX_TOKEN_SIZE, KeystrokeTokenizer.key_labels_length())
            + torch.eye(KeystrokeTokenizer.key_labels_length())[token_ids]
        )
        one_hot_keystrokes = one_hot_keystrokes.view(-1)  # flatten
        return one_hot_keystrokes

    def predict(self, input_keystroke: str) -> bool:
        embedded_input = self._one_hot_encode(input_keystroke)
        embedded_input = embedded_input.to(self._DEVICE)
        self._classifier = self._classifier.to(self._DEVICE)

        with torch.no_grad():
            prediction = self._classifier(embedded_input)
            prediction = torch.argmax(prediction).item()
        return prediction == 1


@deprecated
class IMEDetectorSVM(IMEDetector):
    def __init__(self, svm_model_path: str, tfidf_vectorizer_path: str) -> None:
        super().__init__()
        self.classifiers = None
        self.vectorizer = None
        self.load_model(svm_model_path, tfidf_vectorizer_path)

    def load_model(self, svm_model_path: str, tfidf_vectorizer_path: str) -> None:
        try:
            self.classifiers = joblib.load(svm_model_path)
            print(f"Model loaded from {svm_model_path}")
            self.vectorizer = joblib.load(tfidf_vectorizer_path)
            print(f"Vectorizer loaded from {tfidf_vectorizer_path}")

        except Exception as e:
            print(f"Error loading model and vectorizer.")
            print(e)

    def predict(
        self, input: str, positive_bound: float = 1, neg_bound: float = -0.5
    ) -> bool:
        text_features = self.vectorizer.transform([input])
        predictions = {}
        for label, classifier in self.classifiers.items():
            prediction = classifier.decision_function(text_features)[0]
            predictions[label] = prediction

        if predictions["1"] > positive_bound or (neg_bound < predictions["1"] < 0):
            return True
        else:
            return False

    def predict_eng(
        self, input: str, positive_bound: float = 0.8, neg_bound: float = -0.7
    ) -> bool:
        text_features = self.vectorizer.transform([input])
        predictions = {}
        for label, classifier in self.classifiers.items():
            prediction = classifier.decision_function(text_features)[0]
            predictions[label] = prediction

        if predictions["1"] > positive_bound or (neg_bound < predictions["1"] < 0):
            return True
        else:
            return False

    def predict_positive(self, input: str) -> float:
        text_features = self.vectorizer.transform([input])
        predictions = {}
        for label, classifier in self.classifiers.items():
            prediction = classifier.decision_function(text_features)[0]
            predictions[label] = prediction

        return predictions["1"]


if __name__ == "__main__":
    try:
        my_bopomofo_detector = IMEDetectorOneHot(
            ".\\models\\one_hot_dl_model_bopomofo_2024-10-13.pth",
            device="cuda",
            verbose_mode=True,
        )
        my_eng_detector = IMEDetectorOneHot(
            ".\\models\\one_hot_dl_model_english_2024-10-13.pth",
            device="cuda",
            verbose_mode=True,
        )
        my_cangjie_detector = IMEDetectorOneHot(
            ".\\models\\one_hot_dl_model_cangjie_2024-10-13.pth",
            device="cuda",
            verbose_mode=True,
        )
        my_pinyin_detector = IMEDetectorOneHot(
            ".\\models\\one_hot_dl_model_pinyin_2024-10-13.pth",
            device="cuda",
            verbose_mode=True,
        )
        input_text = "su3cl3"
        while True:
            input_text = input("Enter text: ")
            is_bopomofo = my_bopomofo_detector.predict(input_text)
            # is_cangjie = my_cangjie_detector.predict(input_text)
            is_english = my_eng_detector.predict(input_text)
            is_pinyin = my_pinyin_detector.predict(input_text)

            print(
                Fore.GREEN + "bopomofo" if is_bopomofo else Fore.RED + "bopomofo",
                end=" ",
            )
            # print(
            #     Fore.GREEN + "cangjie" if is_cangjie else Fore.RED + "cangjie", end=" "
            # )
            print(
                Fore.GREEN + "english" if is_english else Fore.RED + "english", end=" "
            )
            print(Fore.GREEN + "pinyin" if is_pinyin else Fore.RED + "pinyin", end=" ")
            print(Style.RESET_ALL)
            print()
    except KeyboardInterrupt:
        print("Exiting...")
