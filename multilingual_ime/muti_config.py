from .ime import BOPOMOFO_IME, CANGJIE_IME, ENGLISH_IME, PINYIN_IME, SPECIAL_IME


class MultiConfig:
    def __init__(self, config_dict: dict = None) -> None:
        if config_dict:
            self._config = self._load_config(config_dict)
        else:
            self._config = {
                "selection_page_size": 5,
                "auto_phrase_learn": False,
                "auto_frequency_learn": False,
                "ime_activation_status": {
                    BOPOMOFO_IME: True,
                    CANGJIE_IME: True,
                    ENGLISH_IME: True,
                    PINYIN_IME: True,
                    SPECIAL_IME: True,
                },
            }

    @property
    def SELECTION_PAGE_SIZE(self) -> int:
        return self._config["selection_page_size"]

    @property
    def AUTO_PHRASE_LEARN(self) -> bool:
        return self._config["auto_phrase_learn"]

    @property
    def AUTO_FREQUENCY_LEARN(self) -> bool:
        return self._config["auto_frequency_learn"]

    @property
    def ACTIVE_IME(self) -> list:
        return [
            ime_name
            for ime_name, status in self._config["ime_activation_status"].items()
            if status
        ]

    def setIMEActivationStatus(self, ime_name: str, status: bool) -> None:
        if ime_name not in self._config["ime_activation_status"]:
            raise ValueError(f"IME {ime_name} not found in config")

        if not isinstance(status, bool):
            raise ValueError(f"status should be a boolean value")

        self._config["ime_activation_status"][ime_name] = status

    def getIMEActivationStatus(self, ime_name: str) -> bool:
        if ime_name not in self._config["ime_activation_status"]:
            raise ValueError(f"IME {ime_name} not found in config")

        return self._config["ime_activation_status"][ime_name]

    def _load_config(self):
        raise NotImplementedError(" _load_config method not implemented")
