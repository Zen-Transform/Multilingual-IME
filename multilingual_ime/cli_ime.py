import time
import keyboard
from colorama import Fore, Style

from multilingual_ime.key_event_handler import KeyEventHandler


def prompt_boolean_setting(question: str, default_value: bool) -> str:
    while True:
        value = input(
            Fore.BLUE
            + question
            + "(default: "
            + (Fore.GREEN + "yes" if default_value else Fore.RED + "no")
            + Fore.BLUE
            + ")(y/n) : "
            + Style.RESET_ALL
        )
        if value == "":
            return default_value
        if value in ["y", "yes"]:
            return True
        if value in ["n", "no"]:
            return False
        print("Invalid value, please input again")


DEFAULT_VERBOSE_MODE = False
DEFAULT_COLOR_MODE = True
DEFAULT_BOPOMOFO_MODE_ENABLED = True
DEFAULT_ENGLISH_MODE_ENABLED = True
DEFAULT_CANGJIE_MODE_ENABLED = False
DEFAULT_PINYIN_MODE_ENABLED = False
DEFAULT_JAPANESE_MODE_ENABLED = True


class CommandLineIME:
    def __init__(self):
        # Print welcome message
        print(
            "-" * 50
            + Fore.GREEN
            + "\nWelcome to the multilingual IME!\n"
            + Fore.BLUE
            + "This is a command line interface (CLI) for the multilingual IME.\n"
            + "You can use this IME to type in multiple languages.\n"
            + "The supported languages are: English, Bopomofo, Cangjie, Pinyin.\n"
            + "If find any bug (possible many XD), please report to us at <https://github.com/Zen-Transform/Multilingual-IME/issues>.\n"
            + Fore.RED
            + "Press ESC to exit\n"
            + Style.RESET_ALL
            + "-" * 50
        )
        # Request settings
        self.skip_all_settings = prompt_boolean_setting(
            "Do you want to skip all settings?", True
        )
        if not self.skip_all_settings:
            self.verbose_mode = prompt_boolean_setting(
                "Do you want to enable verbose mode?", DEFAULT_VERBOSE_MODE
            )
            self.color_mode = prompt_boolean_setting(
                "Do you want to enable color mode?", DEFAULT_COLOR_MODE
            )
            self.bopomofo_mode_enabled = prompt_boolean_setting(
                "Do you want to enable bopomofo mode?", DEFAULT_BOPOMOFO_MODE_ENABLED
            )
            self.english_mode_enabled = prompt_boolean_setting(
                "Do you want to enable english mode?", DEFAULT_ENGLISH_MODE_ENABLED
            )
            self.cangjie_mode_enabled = prompt_boolean_setting(
                "Do you want to enable cangjie mode?", DEFAULT_CANGJIE_MODE_ENABLED
            )
            self.pinyin_mode_enabled = prompt_boolean_setting(
                "Do you want to enable pinyin mode?", DEFAULT_PINYIN_MODE_ENABLED
            )
            self.japanese_mode_enabled = prompt_boolean_setting(
                "Do you want to enable japanese mode?", DEFAULT_JAPANESE_MODE_ENABLED
            )
        else:
            self.verbose_mode = DEFAULT_VERBOSE_MODE
            self.color_mode = DEFAULT_COLOR_MODE
            self.bopomofo_mode_enabled = DEFAULT_BOPOMOFO_MODE_ENABLED
            self.english_mode_enabled = DEFAULT_ENGLISH_MODE_ENABLED
            self.cangjie_mode_enabled = DEFAULT_CANGJIE_MODE_ENABLED
            self.pinyin_mode_enabled = DEFAULT_PINYIN_MODE_ENABLED
            self.japanese_mode_enabled = DEFAULT_JAPANESE_MODE_ENABLED

        # Initialize
        start_time = time.time()
        self.key_event_handler = KeyEventHandler(verbose_mode=self.verbose_mode)
        self.key_event_handler.set_activation_status(
            "bopomofo", self.bopomofo_mode_enabled
        )
        self.key_event_handler.set_activation_status(
            "english", self.english_mode_enabled
        )
        self.key_event_handler.set_activation_status(
            "cangjie", self.cangjie_mode_enabled
        )
        self.key_event_handler.set_activation_status("pinyin", self.pinyin_mode_enabled)
        print("Initialization time: ", time.time() - start_time)
        self._run_timer = None
        self.time_spend = 0
        self.avg_time_spend = 1
        self.key_count = 0

    def update_ui(self):
        print(
            "{:<}\t\t\t{: <5}{: <20}\t{: <5}\t{: <10}\t{: <10}".format(
                self.composition_with_cursor_string,
                self.composition_index_string,
                self.candidate_words_with_cursor_string,
                self.selection_index_string,
                f"Time spend: {self.time_spend:.3f}",
                f"Avg time spend: {self.avg_time_spend:.3f}",
            ),
        )

    def on_key_event(self, event):
        if event.event_type == keyboard.KEY_DOWN:
            self.key_count += 1
            start_time = time.time()

            if event.name in ["enter", "left", "right", "down", "up", "esc"]:
                self.key_event_handler.handle_key(event.name)
            else:
                if keyboard.is_pressed("ctrl") and event.name != "ctrl":
                    self.key_event_handler.handle_key("©" + event.name)
                elif keyboard.is_pressed("shift") and event.name != "shift":
                    self.key_event_handler.handle_key(event.name.upper())
                else:
                    self.key_event_handler.handle_key(event.name)

            self.key_event_handler.slow_handle()
            self.time_spend = time.time() - start_time
            self.avg_time_spend = (
                self.avg_time_spend * (self.key_count - 1) + self.time_spend
            ) / (self.key_count)

        self.update_ui()

    def run(self):
        keyboard.hook(self.on_key_event)
        keyboard.wait("esc")

    @property
    def composition_with_cursor_string(self):
        total_string = []
        total_composition_words = self.key_event_handler.total_composition_words
        freezed_index = self.key_event_handler.freezed_index
        unfreeze_composition_words = self.key_event_handler.unfreeze_composition_words
        composition_index = self.key_event_handler.composition_index

        for i, word in enumerate(total_composition_words):
            if i < freezed_index:
                total_string.append(
                    (Fore.BLUE if self.color_mode else "") + word + Style.RESET_ALL
                )
            elif freezed_index <= i < freezed_index + len(unfreeze_composition_words):
                total_string.append(
                    (Fore.YELLOW if self.color_mode else "") + word + Style.RESET_ALL
                )
            else:
                total_string.append(
                    (Fore.BLUE if self.color_mode else "") + word + Style.RESET_ALL
                )

        total_string.insert(composition_index, Fore.GREEN + "|" + Style.RESET_ALL)
        return "".join(total_string)

    @property
    def candidate_words_with_cursor_string(self):
        if not self.key_event_handler.in_selection_mode:
            return ""

        candidate_word_list = self.key_event_handler.candidate_word_list
        selection_index = self.key_event_handler.selection_index

        output = "["
        for i, word in enumerate(candidate_word_list):
            if i == selection_index:
                output += Fore.GREEN + word + Style.RESET_ALL + " "
            else:
                output += word + " "
        output += "]"
        return output

    @property
    def selection_index_string(self):
        return (
            Fore.GREEN + str(self.key_event_handler.selection_index) + Style.RESET_ALL
            if self.key_event_handler.in_selection_mode
            else ""
        )

    @property
    def composition_index_string(self):
        composition_index = self.key_event_handler.composition_index
        total_composition_words = self.key_event_handler.total_composition_words

        return (
            Fore.GREEN
            + str(len("".join(total_composition_words[:composition_index])))
            + Style.RESET_ALL
        )


if __name__ == "__main__":
    event_wrapper = CommandLineIME()
    event_wrapper.run()
