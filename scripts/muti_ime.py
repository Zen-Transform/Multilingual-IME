import time
import keyboard
import threading
from colorama import Fore, Style

from multilingual_ime.key_event_handler import KeyEventHandler

WITH_COLOR = True
VERBOSE_MODE = False

def get_composition_string_with_cusor(
    total_composition_words: list[str],
    freezed_index: int,
    unfreezed_composition_words: list[int],
    composition_index: int,
) -> str:
    total = []
    for i, word in enumerate(total_composition_words):
        if i < freezed_index:
            total.append((Fore.BLUE if WITH_COLOR else "") + word + Style.RESET_ALL)
        elif freezed_index <= i < freezed_index + len(unfreezed_composition_words):
            total.append((Fore.YELLOW if WITH_COLOR else "") + word + Style.RESET_ALL)
        else:
            total.append((Fore.BLUE if WITH_COLOR else "") + word + Style.RESET_ALL)

    total.insert(composition_index, Fore.GREEN + "|" + Style.RESET_ALL)
    return "".join(total)


def get_candidate_words_with_cursor(
    candidate_word_list: list[str], selection_index: int
) -> str:
    output = "["
    for i, word in enumerate(candidate_word_list):
        if i == selection_index:
            output += Fore.GREEN + word + Style.RESET_ALL + " "
        else:
            output += word + " "
    output += "]"
    return output


class EventWrapper:
    def __init__(self):
        start_time = time.time()
        self.my_keyeventhandler = KeyEventHandler(verbose_mode=VERBOSE_MODE)
        print("Initialization time: ", time.time() - start_time)
        self._run_timer = None

    def update_ui(self):
        print(
            f"{get_composition_string_with_cusor(self.my_keyeventhandler.total_composition_words, self.my_keyeventhandler.freezed_index, self.my_keyeventhandler.unfreezed_composition_words, self.my_keyeventhandler.composition_index)}"
            + f"\t\t {self.my_keyeventhandler.composition_index}"
            + f"\t\t{get_candidate_words_with_cursor(self.my_keyeventhandler.candidate_word_list, self.my_keyeventhandler.selection_index) if self.my_keyeventhandler.in_selection_mode else ''}"
            + f"\t\t{self.my_keyeventhandler.selection_index if self.my_keyeventhandler.in_selection_mode else ''}"
        )

    def slow_handle(self):
        self.my_keyeventhandler.slow_handle()
        self.update_ui()

    def on_key_event(self, event):
        if event.event_type == keyboard.KEY_DOWN:
            if self._run_timer is not None:
                self._run_timer.cancel()

            if event.name in ["enter", "left", "right", "down", "up", "esc"]:
                self.my_keyeventhandler.handle_key(event.name)
            else:
                if keyboard.is_pressed("ctrl") and event.name != "ctrl":
                    self.my_keyeventhandler.handle_key("©" + event.name)
                elif keyboard.is_pressed("shift") and event.name != "shift":
                    self.my_keyeventhandler.handle_key(event.name.upper())
                else:
                    self.my_keyeventhandler.handle_key(event.name)

                self._run_timer = threading.Timer(0.25, self.slow_handle)
                self._run_timer.start()

        self.update_ui()

    def run(self):
        keyboard.hook(self.on_key_event)
        keyboard.wait("esc")


if __name__ == "__main__":
    event_wrapper = EventWrapper()
    event_wrapper.run()
