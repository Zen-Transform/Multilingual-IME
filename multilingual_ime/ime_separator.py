from pathlib import Path

from .ime_detector import IMEDetectorOneHot


class IMESeparator:
    def __init__(self, use_cuda: bool = True) -> None:
        self._DEVICE = "cuda" if use_cuda else "cpu"

        self._bopomofo_detector = IMEDetectorOneHot(
            Path(__file__).parent
            / "src"
            / "model_dump"
            / "one_hot_dl_model_bopomofo_2024-07-26.pkl",
            device=self._DEVICE,
        )
        self._eng_detector = IMEDetectorOneHot(
            Path(__file__).parent
            / "src"
            / "model_dump"
            / "one_hot_dl_model_english_2024-07-26.pkl",
            device=self._DEVICE,
        )
        self._cangjie_detector = IMEDetectorOneHot(
            Path(__file__).parent
            / "src"
            / "model_dump"
            / "one_hot_dl_model_cangjie_2024-07-26.pkl",
            device=self._DEVICE,
        )
        self._pinyin_detector = IMEDetectorOneHot(
            Path(__file__).parent
            / "src"
            / "model_dump"
            / "one_hot_dl_model_pinyin_2024-07-26.pkl",
            device=self._DEVICE,
        )

    def separate(self, input_stroke: str) -> list[list[(str, str)]]:
        results = []
        detector_groups = [
            (self._bopomofo_detector, "bopomofo"),
            (self._cangjie_detector, "cangjie"),
            (self._eng_detector, "english"),
            (self._pinyin_detector, "pinyin"),
        ]

        for index in range(0, len(input_stroke)):
            former_keystrokes = input_stroke[:index]
            latter_keystrokes = input_stroke[index:]
            for former_detector, former_language in detector_groups:
                for latter_detector, latter_language in detector_groups:
                    if (
                        former_detector.predict(former_keystrokes)
                        and latter_detector.predict(latter_keystrokes)
                        and former_detector != latter_detector
                    ):
                        if former_keystrokes == "":
                            results.append([(latter_language, latter_keystrokes)])
                        elif latter_keystrokes == "":
                            results.append([(former_language, former_keystrokes)])
                        else:
                            results.append(
                                [
                                    (former_language, former_keystrokes),
                                    (latter_language, latter_keystrokes),
                                ]
                            )

        results.append(
            [("english", input_stroke)]
        )  # Assume User intends to input English(gibberish)

        return results


if __name__ == "__main__":
    my_separator = IMESeparator(use_cuda=False)
    input_text = "su3cl3goodnight"
    print(my_separator.separate(input_text))
