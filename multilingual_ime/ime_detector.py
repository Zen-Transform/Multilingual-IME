import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from colorama import Fore, Style

from .keystroke_tokenizer import KeystrokeTokenizer
from .deep_models import LanguageDetectorModel, TokenDetectorModel

MAX_TOKEN_SIZE = 30


class IMEDetector(ABC):
    """
    Abstract base class for IME detectors.
    This class defines the interface for IME detectors that can be used to
    detect whether a given keystroke is valid for a specific IME.
    """

    @abstractmethod
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def load_model(self, model_path: str | Path) -> None:
        """
        Load the model from the specified path.
        Args:
            model_path (str | Path): The path to the model file.
        """

    @abstractmethod
    def predict(self, input_keystroke: str) -> bool:
        """
        Predict whether the given keystroke is valid for the IME.
        Args:
            input_keystroke (str): The keystroke to be checked.
        Returns:
            bool: True if the keystroke is valid for the IME, False otherwise.
        """


class IMEDetectorOneHot(IMEDetector):
    """
    IMEDetectorOneHot is a detector that uses a one-hot encoding NN model
    to detect whether a given keystrokes is a valid to a specific IME.
    It is designed to work with a pre-trained model that is loaded from a specified path.
    """

    def __init__(
        self, model_path: str, device: str = "cuda", verbose_mode: bool = False
    ) -> None:
        super().__init__()
        self.logger.setLevel(logging.INFO if verbose_mode else logging.WARNING)
        self._classifier = None
        self._device = device

        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("cuda is not available, using cpu instead")
            self._device = "cpu"
        if isinstance(model_path, Path):
            model_path = str(model_path)
        if not model_path.endswith(".pth"):
            self.logger.error("Invalid model path. Model must be a .pth file.")
            return

        self.load_model(model_path)
        self.logger.info(
            f"Detector created using the {self._device} device." if verbose_mode else ""
        )

    def load_model(self, model_path: str | Path) -> None:
        try:
            self._classifier = LanguageDetectorModel(
                input_shape=MAX_TOKEN_SIZE * KeystrokeTokenizer.key_labels_length(),
                num_classes=2,
            )
            self._classifier.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            self.logger.info("Model loaded from %s", model_path)
            self.logger.info(self._classifier)
        except (FileNotFoundError, RuntimeError, OSError) as e:
            self.logger.error("Error loading model %s", model_path)
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
        embedded_input = embedded_input.to(self._device)
        if self._classifier is not None:
            self._classifier = self._classifier.to(self._device)
        else:
            self.logger.error("Classifier model is not loaded.")
            return False

        with torch.no_grad():
            prediction = self._classifier(embedded_input)
            prediction = torch.argmax(prediction).item()
        return prediction == 1


class IMETokenDetectorDL(IMEDetector):
    """
    IMEDetectorOneHot is a token detector that uses a one-hot encoding NN model
    to detect whether a given keystrokes is a token of a specific language.
    It is designed to work with a pre-trained model that is loaded from a specified path.
    """

    def __init__(
        self, model_path: str | Path, device: str = "cuda", verbose_mode: bool = False
    ) -> None:
        super().__init__()
        self.logger.setLevel(logging.INFO if verbose_mode else logging.WARNING)
        self._classifier = None
        self._device = device

        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("cuda is not available, using cpu instead")
            self._device = "cpu"
        if isinstance(model_path, Path):
            model_path = str(model_path)
        if not model_path.endswith(".pth"):
            self.logger.error("Invalid model path. Model must be a .pth file.")
            return

        self.load_model(model_path)
        self.logger.info(
            f"Detector created using the {self._device} device." if verbose_mode else ""
        )

    def load_model(self, model_path: str | Path) -> None:
        try:
            self._classifier = TokenDetectorModel(
                input_shape=MAX_TOKEN_SIZE * KeystrokeTokenizer.key_labels_length(),
                num_classes=1,  # only 1 class for token detection
            )
            self._classifier.load_state_dict(
                torch.load(model_path, map_location=self._device, weights_only=True)
            )
            self.logger.info("Model loaded from %s", model_path)
            self.logger.info(self._classifier)
        except (FileNotFoundError, RuntimeError, OSError) as e:
            self.logger.error("Error loading model %s", model_path)
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
        embedded_input = embedded_input.to(self._device)
        if self._classifier is not None:
            self._classifier = self._classifier.to(self._device)
        else:
            self.logger.error("Classifier model is not loaded.")
            return False

        with torch.no_grad():
            prediction = self._classifier(embedded_input)
            prediction = torch.round(prediction).item()
        return prediction == 1


if __name__ == "__main__":
    try:
        my_bopomofo_detector = IMETokenDetectorDL(
            ".\\models\\one_hot_dl_token_model_bopomofo_2024-10-27.pth",
            device="cuda",
            verbose_mode=True,
        )
        my_eng_detector = IMETokenDetectorDL(
            ".\\models\\one_hot_dl_token_model_english_2024-10-27.pth",
            device="cuda",
            verbose_mode=True,
        )
        my_cangjie_detector = IMETokenDetectorDL(
            ".\\models\\one_hot_dl_token_model_cangjie_2024-10-27.pth",
            device="cuda",
            verbose_mode=True,
        )
        my_pinyin_detector = IMETokenDetectorDL(
            ".\\models\\one_hot_dl_token_model_pinyin_2024-10-27.pth",
            device="cuda",
            verbose_mode=True,
        )
        while True:
            input_text = input("Enter text: ")
            is_bopomofo = my_bopomofo_detector.predict(input_text)
            is_cangjie = my_cangjie_detector.predict(input_text)
            is_english = my_eng_detector.predict(input_text)
            is_pinyin = my_pinyin_detector.predict(input_text)

            print(
                Fore.GREEN + "bopomofo" if is_bopomofo else Fore.RED + "bopomofo",
                end=" ",
            )
            print(
                Fore.GREEN + "cangjie" if is_cangjie else Fore.RED + "cangjie", end=" "
            )
            print(
                Fore.GREEN + "english" if is_english else Fore.RED + "english", end=" "
            )
            print(Fore.GREEN + "pinyin" if is_pinyin else Fore.RED + "pinyin", end=" ")
            print(Style.RESET_ALL)
            print()
    except KeyboardInterrupt:
        print("Exiting...")
